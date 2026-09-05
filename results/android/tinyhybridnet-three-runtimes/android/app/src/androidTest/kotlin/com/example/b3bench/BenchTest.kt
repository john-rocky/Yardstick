package com.example.b3bench

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.os.Build
import android.os.SystemClock
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.EValue
import org.pytorch.executorch.ExecuTorchRuntime
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs

/**
 * Same TinyHybridNet (conv stem + one transformer block, 1x3x224x224 -> 10 logits), same frozen weights,
 * three on-device runtimes, one device, one process. For every configuration: 5 warm-up runs, then 30 timed
 * runs of (write input, run, read output); reports median and min wall time and the max |diff| vs the
 * PyTorch golden output on the fixed golden input.
 */
@RunWith(AndroidJUnit4::class)
class BenchTest {
    private val target get() = InstrumentationRegistry.getInstrumentation().targetContext
    private val results = JSONArray()

    private fun readFloats(name: String): FloatArray {
        val bytes = target.assets.open(name).use { it.readBytes() }
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(fb.remaining()).also { fb.get(it) }
    }

    private fun assetToFile(name: String): String {
        val f = File(target.filesDir, name)
        target.assets.open(name).use { src -> f.outputStream().use { dst -> src.copyTo(dst) } }
        return f.absolutePath
    }

    private fun record(runtime: String, backend: String, golden: FloatArray, expected: FloatArray, note: String, run: (FloatArray) -> FloatArray) {
        try {
            val first = run(golden)
            var maxAbs = 0f
            for (i in expected.indices) maxAbs = maxOf(maxAbs, abs(first[i] - expected[i]))
            val argmaxOk = expected.indices.maxBy { expected[it] } == first.indices.maxBy { first[it] }
            repeat(WARMUP) { run(golden) }
            val times = DoubleArray(RUNS) {
                val t0 = SystemClock.elapsedRealtimeNanos()
                run(golden)
                (SystemClock.elapsedRealtimeNanos() - t0) / 1e6
            }
            times.sort()
            val median = if (RUNS % 2 == 0) (times[RUNS / 2 - 1] + times[RUNS / 2]) / 2 else times[RUNS / 2]
            val o = JSONObject().put("runtime", runtime).put("backend", backend).put("median_ms", median)
                .put("min_ms", times[0]).put("p90_ms", times[(RUNS * 0.9).toInt()]).put("max_abs_diff", maxAbs.toDouble())
                .put("argmax_ok", argmaxOk).put("note", note).put("runs", RUNS)
            Log.i(TAG, "B3RESULT $o")
            results.put(o)
        } catch (t: Throwable) {
            val o = JSONObject().put("runtime", runtime).put("backend", backend).put("error", t.toString()).put("note", note)
            Log.w(TAG, "B3RESULT $o", t)
            results.put(o)
        }
        SystemClock.sleep(1500)
    }

    // ---- ExecuTorch ------------------------------------------------------------------------------------
    private fun executorch(pteAsset: String, backendLabel: String, numThreads: Int, golden: FloatArray, expected: FloatArray) {
        val path = assetToFile(pteAsset)
        var module: Module? = null
        record("ExecuTorch 1.4.0", backendLabel, golden, expected, "numThreads=$numThreads; registered=${ExecuTorchRuntime.getRegisteredBackends().toList()}") { input ->
            val m = module ?: Module.load(path, Module.LOAD_MODE_MMAP, numThreads).also { it.loadMethod("forward"); module = it }
            val out = m.forward(EValue.from(Tensor.fromBlob(input, SHAPE)))
            out[0].toTensor().dataAsFloatArray
        }
        module?.destroy()
    }

    // ---- LiteRT ----------------------------------------------------------------------------------------
    private fun litert(options: CompiledModel.Options, backendLabel: String, note: String, golden: FloatArray, expected: FloatArray) {
        var model: CompiledModel? = null
        var inputs: List<com.google.ai.edge.litert.TensorBuffer>? = null
        var outputs: List<com.google.ai.edge.litert.TensorBuffer>? = null
        record("LiteRT 2.2.0", backendLabel, golden, expected, note) { input ->
            val m = model ?: CompiledModel.create(target.assets, "tinyhybridnet.tflite", options).also {
                model = it; inputs = it.createInputBuffers(); outputs = it.createOutputBuffers()
            }
            inputs!![0].writeFloat(input)
            m.run(inputs!!, outputs!!)
            outputs!![0].readFloat()
        }
        inputs?.forEach { it.close() }; outputs?.forEach { it.close() }; model?.close()
    }

    // ---- ONNX Runtime ----------------------------------------------------------------------------------
    private fun ort(backendLabel: String, configure: (OrtSession.SessionOptions) -> Unit, note: String, golden: FloatArray, expected: FloatArray) {
        val env = OrtEnvironment.getEnvironment()
        val bytes = target.assets.open("tinyhybridnet.onnx").use { it.readBytes() }
        var session: OrtSession? = null
        record("ONNX Runtime 1.24.3", backendLabel, golden, expected, note) { input ->
            val s = session ?: OrtSession.SessionOptions().let { o -> configure(o); env.createSession(bytes, o) }.also { session = it }
            OnnxTensor.createTensor(env, FloatBuffer.wrap(input), SHAPE).use { t ->
                s.run(mapOf("input" to t)).use { r ->
                    @Suppress("UNCHECKED_CAST")
                    (r[0].value as Array<FloatArray>)[0]
                }
            }
        }
        session?.close()
    }

    @Test
    fun benchAll() {
        val golden = readFloats("golden_input_1x3x224x224_f32.bin")
        val expected = readFloats("golden_output_1x10_f32.bin")
        val soc = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL else "?"
        val device = JSONObject().put("model", Build.MODEL).put("soc", soc).put("android", Build.VERSION.RELEASE)
            .put("sdk", Build.VERSION.SDK_INT).put("cores", Runtime.getRuntime().availableProcessors()).put("abi", Build.SUPPORTED_ABIS.joinToString(","))
        Log.i(TAG, "B3DEVICE $device")

        executorch("tinyhybridnet_xnnpack.pte", "XNNPACK (default threads)", 0, golden, expected)
        executorch("tinyhybridnet_xnnpack.pte", "XNNPACK (4 threads)", 4, golden, expected)
        executorch("tinyhybridnet_portable.pte", "portable kernels, no delegate", 0, golden, expected)

        litert(CompiledModel.Options(Accelerator.CPU), "CPU / XNNPACK (default threads)", "CompiledModel, Accelerator.CPU", golden, expected)
        litert(CompiledModel.Options(Accelerator.CPU).apply { cpuOptions = CompiledModel.CpuOptions(4, null, null) }, "CPU / XNNPACK (4 threads)", "CompiledModel, CpuOptions(numThreads=4)", golden, expected)
        litert(CompiledModel.Options(Accelerator.GPU), "GPU", "CompiledModel, Accelerator.GPU", golden, expected)

        ort("CPU EP (default threads)", {}, "SessionOptions default", golden, expected)
        ort("XNNPACK EP (default)", { it.addXnnpack(emptyMap()) }, "addXnnpack(emptyMap())", golden, expected)
        ort("XNNPACK EP (4 threads)", { it.addXnnpack(mapOf("intra_op_num_threads" to "4")) }, "addXnnpack(intra_op_num_threads=4)", golden, expected)
        ort("NNAPI EP", { it.addNnapi() }, "addNnapi()", golden, expected)

        val out = JSONObject().put("device", device).put("results", results)
        val f = File(target.getExternalFilesDir(null), "b3_results.json")
        f.writeText(out.toString(2))
        Log.i(TAG, "B3DONE wrote ${f.absolutePath}")
    }

    companion object {
        private const val TAG = "B3Bench"
        private const val WARMUP = 5
        private const val RUNS = 30
        private val SHAPE = longArrayOf(1, 3, 224, 224)
    }
}
