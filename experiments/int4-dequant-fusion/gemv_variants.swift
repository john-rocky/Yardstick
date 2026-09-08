// gemv_variants.swift — what limits a K-sliced int4 decode GEMV on Apple GPUs?
//
// Kernel L is an int4 GEMV in the K-sliced, threadgroup-reduction style common to WGSL
// delegates: weights stored as [K/4][N/4] uint2 (one uint2 = 4 K-values x 4 output channels of
// 4-bit nibbles), per-group (32 K) half4 scale and half4 zero, threadgroup 16 channel-groups x
// 16 K-slices (256 threads), nibbles unpacked with fp16 arithmetic (half(byte)*0.0625 -> floor /
// frac*16), 16 partial sums reduced through threadgroup memory with 4 barrier rounds. Its
// whole-token time matches the weight-GEMV family measured inside LiteRT-LM 0.17.0 by dispatch
// ablation within 4-12%, which is what makes it a usable baseline. Variants change ONE thing each:
//   L_int  : integer nibble extraction (& 15, >> 4) instead of fp16 arithmetic
//   L_u4   : two K-steps per 16-byte uint4 load instead of one uint2 (8 B) per step
//   L_ks8  : 8 K-slices x 32 channel-groups per threadgroup (wider coalesced rows, fewer partials)
//   L_nozp : no zero-point term (symmetric quant: only a scale per group)
//   C      : row-major [N][K] simdgroup GEMV with in-register integer unpack (gemv_bench.swift)
// Whole-token timing: 36 layers x 7 matrices of Qwen3-4B, one command buffer, GPU-side time.
//   swiftc -O gemv_variants.swift -o gemv_variants && ./gemv_variants [iters]
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;

// K-sliced layout: W[k4 * N4 + n4] (uint2 = 4 K-values x 4 channels of nibbles),
// scales/zeros [g * N4 + n4] (half4 per 32-K group per 4 channels).
inline void unpack_ftrick(uint2 w, thread half4 &v0, thread half4 &v1, thread half4 &v2, thread half4 &v3) {
  half4 lo = half4(half(w.x & 255u), half((w.x >> 8u) & 255u), half((w.x >> 16u) & 255u), half((w.x >> 24u) & 255u)) * 0.0625h;
  half4 hi = half4(half(w.y & 255u), half((w.y >> 8u) & 255u), half((w.y >> 16u) & 255u), half((w.y >> 24u) & 255u)) * 0.0625h;
  v0.y = floor(lo.x); v0.w = floor(lo.y); v0.x = (lo.x - v0.y) * 16.0h; v0.z = (lo.y - v0.w) * 16.0h;
  v1.y = floor(lo.z); v1.w = floor(lo.w); v1.x = (lo.z - v1.y) * 16.0h; v1.z = (lo.w - v1.w) * 16.0h;
  v2.y = floor(hi.x); v2.w = floor(hi.y); v2.x = (hi.x - v2.y) * 16.0h; v2.z = (hi.y - v2.w) * 16.0h;
  v3.y = floor(hi.z); v3.w = floor(hi.w); v3.x = (hi.z - v3.y) * 16.0h; v3.z = (hi.w - v3.w) * 16.0h;
}
inline void unpack_int(uint2 w, thread half4 &v0, thread half4 &v1, thread half4 &v2, thread half4 &v3) {
  v0 = half4(half(w.x & 15u), half((w.x >> 4u) & 15u), half((w.x >> 8u) & 15u), half((w.x >> 12u) & 15u));
  v1 = half4(half((w.x >> 16u) & 15u), half((w.x >> 20u) & 15u), half((w.x >> 24u) & 15u), half(w.x >> 28u));
  v2 = half4(half(w.y & 15u), half((w.y >> 4u) & 15u), half((w.y >> 8u) & 15u), half((w.y >> 12u) & 15u));
  v3 = half4(half((w.y >> 16u) & 15u), half((w.y >> 20u) & 15u), half((w.y >> 24u) & 15u), half(w.y >> 28u));
}
constant int KS [[function_constant(0)]];      // K-slices per threadgroup (threads.y)
constant int MODE [[function_constant(1)]];    // 0 = fp16-arithmetic unpack, 1 = integer unpack
constant int ZP [[function_constant(2)]];      // 1 = scale + zero-point, 0 = scale only
constant int U4 [[function_constant(3)]];      // 1 = 16-byte loads (2 K-steps), W in [K/8][N/4] uint4 layout

kernel void gemv_lrt(device const uint2* W [[buffer(0)]], device const half4* scales [[buffer(1)]],
                     device const half4* zeros [[buffer(2)]], device const half4* x [[buffer(3)]],
                     device half4* y [[buffer(4)]], constant uint& K [[buffer(5)]], constant uint& N [[buffer(6)]],
                     uint3 tid [[thread_position_in_threadgroup]], uint3 gid [[threadgroup_position_in_grid]]) {
  threadgroup half4 red[256];
  const int CG = 256 / KS;
  uint N4 = N / 4, G = K / 32; uint n4 = gid.x * CG + tid.x;
  half4 acc = half4(0.0h);
  if (n4 < N4) {
    for (uint g = tid.y; g < G; g += KS) {
      half4 s = scales[g * N4 + n4];
      half4 z = ZP ? (-s * (half4(8.0h) + zeros[g * N4 + n4])) : half4(0.0h);
      if (U4) {
        device const uint4* W4 = (device const uint4*)W;
        for (uint i = 0; i < 8; i += 2) {
          uint k4 = g * 8 + i;
          uint4 w2 = W4[(ulong)(k4 / 2) * N4 + n4];
          uint2 wa = uint2(w2.x, w2.y), wb = uint2(w2.z, w2.w); half4 v0, v1, v2, v3;
          if (MODE == 0) unpack_ftrick(wa, v0, v1, v2, v3); else unpack_int(wa, v0, v1, v2, v3);
          half4 xv = x[k4];
          v0 = fma(v0, s, z); v1 = fma(v1, s, z); v2 = fma(v2, s, z); v3 = fma(v3, s, z);
          acc = fma(half4(xv.x), v0, acc); acc = fma(half4(xv.y), v1, acc); acc = fma(half4(xv.z), v2, acc); acc = fma(half4(xv.w), v3, acc);
          if (MODE == 0) unpack_ftrick(wb, v0, v1, v2, v3); else unpack_int(wb, v0, v1, v2, v3);
          xv = x[k4 + 1];
          v0 = fma(v0, s, z); v1 = fma(v1, s, z); v2 = fma(v2, s, z); v3 = fma(v3, s, z);
          acc = fma(half4(xv.x), v0, acc); acc = fma(half4(xv.y), v1, acc); acc = fma(half4(xv.z), v2, acc); acc = fma(half4(xv.w), v3, acc);
        }
      } else {
        for (uint i = 0; i < 8; i++) {
          uint k4 = g * 8 + i;
          uint2 w = W[(ulong)k4 * N4 + n4]; half4 v0, v1, v2, v3;
          if (MODE == 0) unpack_ftrick(w, v0, v1, v2, v3); else unpack_int(w, v0, v1, v2, v3);
          half4 xv = x[k4];
          v0 = fma(v0, s, z); v1 = fma(v1, s, z); v2 = fma(v2, s, z); v3 = fma(v3, s, z);
          acc = fma(half4(xv.x), v0, acc); acc = fma(half4(xv.y), v1, acc); acc = fma(half4(xv.z), v2, acc); acc = fma(half4(xv.w), v3, acc);
        }
      }
    }
  }
  red[tid.x * KS + tid.y] = acc;
  for (int st = KS / 2; st > 0; st /= 2) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid.y) < st) { acc += red[tid.x * KS + tid.y + st]; red[tid.x * KS + tid.y] = acc; }
  }
  if (tid.y == 0 && n4 < N4) y[n4] = acc;
}

// row-major [N][K] simdgroup GEMV with the fp16-arithmetic unpack
kernel void gemv_C_ftrick(device const uint2* packed [[buffer(0)]], device const half* scales [[buffer(1)]],
                   device const half* zeros [[buffer(2)]], device const half4* x [[buffer(3)]],
                   device half* y [[buffer(4)]], constant uint& K [[buffer(5)]], constant uint& N [[buffer(6)]],
                   uint tgid [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                   uint simd_id [[simdgroup_index_in_threadgroup]]) {
  uint n = tgid * 4 + simd_id; if (n >= N) return;
  uint per_row = K / 16; float acc = 0.0f;
  for (uint k16 = simd_lane; k16 < per_row; k16 += 32) {
    uint2 v = packed[n * per_row + k16]; uint g = n * (K / 32) + k16 / 2;
    half s = scales[g]; half z = -s * (8.0h + zeros[g]);
    half4 v0, v1, v2, v3; unpack_ftrick(v, v0, v1, v2, v3);   // (lo0,hi0,lo1,hi1) ordering: 16 consecutive K values, any fixed order is fine for a benchmark
    uint xo = k16 * 4; half4 xs = x[xo] + x[xo + 1] + x[xo + 2] + x[xo + 3];
    float wx = float(dot(v0, x[xo]) + dot(v1, x[xo + 1]) + dot(v2, x[xo + 2]) + dot(v3, x[xo + 3]));
    acc += float(s) * wx + float(z) * float(xs.x + xs.y + xs.z + xs.w);
  }
  acc = simd_sum(acc);
  if (simd_lane == 0) y[n] = half(acc);
}

// reference: row-major [N][K] simdgroup GEMV, integer unpack (kernel C of gemv_bench)
kernel void gemv_C(device const uint2* packed [[buffer(0)]], device const half* scales [[buffer(1)]],
                   device const half* zeros [[buffer(2)]], device const half4* x [[buffer(3)]],
                   device half* y [[buffer(4)]], constant uint& K [[buffer(5)]], constant uint& N [[buffer(6)]],
                   uint tgid [[threadgroup_position_in_grid]], uint simd_lane [[thread_index_in_simdgroup]],
                   uint simd_id [[simdgroup_index_in_threadgroup]]) {
  uint n = tgid * 4 + simd_id; if (n >= N) return;
  uint per_row = K / 16; float acc = 0.0f;
  for (uint k16 = simd_lane; k16 < per_row; k16 += 32) {
    uint2 v = packed[n * per_row + k16]; uint g = n * (K / 32) + k16 / 2;
    half s = scales[g]; half z = -s * (8.0h + zeros[g]);
    half4 w0 = half4(half(v.x & 15u), half((v.x >> 4u) & 15u), half((v.x >> 8u) & 15u), half((v.x >> 12u) & 15u));
    half4 w1 = half4(half((v.x >> 16u) & 15u), half((v.x >> 20u) & 15u), half((v.x >> 24u) & 15u), half(v.x >> 28u));
    half4 w2 = half4(half(v.y & 15u), half((v.y >> 4u) & 15u), half((v.y >> 8u) & 15u), half((v.y >> 12u) & 15u));
    half4 w3 = half4(half((v.y >> 16u) & 15u), half((v.y >> 20u) & 15u), half((v.y >> 24u) & 15u), half(v.y >> 28u));
    uint xo = k16 * 4; half4 xs = x[xo] + x[xo + 1] + x[xo + 2] + x[xo + 3];
    float wx = float(dot(w0, x[xo]) + dot(w1, x[xo + 1]) + dot(w2, x[xo + 2]) + dot(w3, x[xo + 3]));
    acc += float(s) * wx + float(z) * float(xs.x + xs.y + xs.z + xs.w);
  }
  acc = simd_sum(acc);
  if (simd_lane == 0) y[n] = half(acc);
}
"""

let dev = MTLCreateSystemDefaultDevice()!
let lib = try! dev.makeLibrary(source: src, options: nil)
let q = dev.makeCommandQueue()!
func pso(_ n: String) -> MTLComputePipelineState { try! dev.makeComputePipelineState(function: lib.makeFunction(name: n)!) }
func psoL(ks: Int32, mode: Int32, zp: Int32, u4: Int32) -> MTLComputePipelineState {
  let c = MTLFunctionConstantValues(); var ks = ks, mode = mode, zp = zp, u4 = u4
  c.setConstantValue(&ks, type: .int, index: 0); c.setConstantValue(&mode, type: .int, index: 1); c.setConstantValue(&zp, type: .int, index: 2); c.setConstantValue(&u4, type: .int, index: 3)
  return try! dev.makeComputePipelineState(function: try! lib.makeFunction(name: "gemv_lrt", constantValues: c))
}
let iters = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 20
func buf(_ bytes: Int) -> MTLBuffer { dev.makeBuffer(length: bytes, options: .storageModeShared)! }
func fill(_ b: MTLBuffer, _ seed: UInt32) { let p = b.contents().assumingMemoryBound(to: UInt32.self); var s = seed &+ 1; for i in 0..<(b.length / 4) { s = s &* 1664525 &+ 1013904223; p[i] = s } }
func fillHalf(_ b: MTLBuffer, _ v: Float) { let p = b.contents().assumingMemoryBound(to: UInt16.self); let bits = Float16(v).bitPattern; for i in 0..<(b.length / 2) { p[i] = bits } }
func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
func timeGPU(_ enc: (MTLComputeCommandEncoder) -> Void) -> Double {
  let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!; enc(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  return cb.gpuEndTime - cb.gpuStartTime
}

struct Mat { let K: Int; let N: Int; let w: MTLBuffer; let s: MTLBuffer; let z: MTLBuffer; let y: MTLBuffer }   // LiteRT layout
let modelName = CommandLine.arguments.contains("--ds") ? "DeepSeek-R1-Distill-Qwen-1.5B (28 layers)" : "Qwen3-4B (36 layers)"
let layerShapes = CommandLine.arguments.contains("--ds")
  ? [(1536, 8960), (1536, 8960), (8960, 1536), (1536, 1536), (1536, 1536), (1536, 256), (1536, 256)]
  : [(2560, 9728), (2560, 9728), (9728, 2560), (2560, 4096), (4096, 2560), (2560, 1024), (2560, 1024)]
let nLayers = CommandLine.arguments.contains("--ds") ? 28 : 36
var layers: [[Mat]] = []
for l in 0..<nLayers {
  layers.append(layerShapes.enumerated().map { (i, kn) in
    let (K, N) = kn; let m = Mat(K: K, N: N, w: buf(K / 2 * N), s: buf((K / 32) * (N / 4) * 8), z: buf((K / 32) * (N / 4) * 8), y: buf(N * 2))
    fill(m.w, UInt32(l * 7 + i)); fillHalf(m.s, 0.01); fillHalf(m.z, 7.5); return m })
}
let x = buf(9728 * 2); fillHalf(x, 0.5)
var wb = 0, sb = 0
for layer in layers { for m in layer { wb += m.K / 2 * m.N; sb += (m.K / 32) * (m.N / 4) * 8 } }
let weightBytes = Double(wb), scaleBytes = Double(sb)

func runToken(_ name: String, _ p: MTLComputePipelineState, tpg: MTLSize, groups: (Int, Int) -> Int, rowMajor: Bool = false) {
  var ts: [Double] = []
  for _ in 0..<(iters + 3) {
    ts.append(timeGPU { e in
      e.setComputePipelineState(p)
      for layer in layers { for m in layer {
        var k = UInt32(m.K), n = UInt32(m.N)
        e.setBuffer(m.w, offset: 0, index: 0); e.setBuffer(m.s, offset: 0, index: 1); e.setBuffer(m.z, offset: 0, index: 2); e.setBuffer(x, offset: 0, index: 3); e.setBuffer(m.y, offset: 0, index: 4)
        e.setBytes(&k, length: 4, index: 5); e.setBytes(&n, length: 4, index: 6)
        e.dispatchThreadgroups(MTLSize(width: groups(m.K, m.N), height: 1, depth: 1), threadsPerThreadgroup: tpg)
      } }
    })
  }
  ts.removeFirst(3); let t = median(ts)
  let ms = t * 1e3
  let gbsAll = (weightBytes + 2.0 * scaleBytes) / t / 1e9
  let gbsW = weightBytes / t / 1e9
  let padded = name + String(repeating: " ", count: max(0, 11 - name.count))
  print(padded + String(format: "%7.2f ms/token (%d layers x 7 GEMVs)   %4.0f GB/s int4+scale+zero   %4.0f GB/s int4 only", ms, nLayers, gbsAll, gbsW))
}
let wGB = String(format: "%.3f", weightBytes / 1e9), sGB = String(format: "%.3f", scaleBytes / 1e9)
print("device \(dev.name)  iters \(iters)  model \(modelName)  layer weights int4 \(wGB) GB + scales \(sGB) GB + zeros \(sGB) GB per token")
let tpg16 = MTLSize(width: 16, height: 16, depth: 1)
runToken("L(base)", psoL(ks: 16, mode: 0, zp: 1, u4: 0), tpg: tpg16, groups: { _, n in (n / 4 + 15) / 16 })
runToken("L_int", psoL(ks: 16, mode: 1, zp: 1, u4: 0), tpg: tpg16, groups: { _, n in (n / 4 + 15) / 16 })
runToken("L_nozp", psoL(ks: 16, mode: 0, zp: 0, u4: 0), tpg: tpg16, groups: { _, n in (n / 4 + 15) / 16 })
runToken("L_u4", psoL(ks: 16, mode: 0, zp: 1, u4: 1), tpg: tpg16, groups: { _, n in (n / 4 + 15) / 16 })
runToken("L_int_u4", psoL(ks: 16, mode: 1, zp: 1, u4: 1), tpg: tpg16, groups: { _, n in (n / 4 + 15) / 16 })
runToken("L_ks8", psoL(ks: 8, mode: 0, zp: 1, u4: 0), tpg: MTLSize(width: 32, height: 8, depth: 1), groups: { _, n in (n / 4 + 31) / 32 })
runToken("L_ks4", psoL(ks: 4, mode: 0, zp: 1, u4: 0), tpg: MTLSize(width: 64, height: 4, depth: 1), groups: { _, n in (n / 4 + 63) / 64 })
runToken("L_int_ks8", psoL(ks: 8, mode: 1, zp: 1, u4: 0), tpg: MTLSize(width: 32, height: 8, depth: 1), groups: { _, n in (n / 4 + 31) / 32 })
runToken("L_ks32", psoL(ks: 32, mode: 0, zp: 1, u4: 0), tpg: MTLSize(width: 8, height: 32, depth: 1), groups: { _, n in (n / 4 + 7) / 8 })
runToken("L_ks64", psoL(ks: 64, mode: 0, zp: 1, u4: 0), tpg: MTLSize(width: 4, height: 64, depth: 1), groups: { _, n in (n / 4 + 3) / 4 })
runToken("C(simd)", pso("gemv_C"), tpg: MTLSize(width: 128, height: 1, depth: 1), groups: { _, n in (n + 3) / 4 }, rowMajor: true)
runToken("C_ftrick", pso("gemv_C_ftrick"), tpg: MTLSize(width: 128, height: 1, depth: 1), groups: { _, n in (n + 3) / 4 }, rowMajor: true)
runToken("L(again)", psoL(ks: 16, mode: 0, zp: 1, u4: 0), tpg: tpg16, groups: { _, n in (n / 4 + 15) / 16 })
