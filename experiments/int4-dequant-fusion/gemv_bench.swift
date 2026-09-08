// gemv_bench.swift — does fusing int4 dequant into the GEMV pay on Apple GPUs?
//
// Models a two-pass weight path ([A] dequantize the int4 blockwise (gs32) weight matrix into an
// fp16 device buffer, then [B] an fp16 GEMV consumes it) and compares it with [C] a single
// fused int4 GEMV that dequantizes in registers. NOTE: steady-state decode in LiteRT-LM does not
// run a two-pass path (see the doc); this file only shows what such a path would cost. Same
// matrices as Qwen3-4B decode
// (K x N: 2560x9728 gate/up, 9728x2560 down, 2560x4096 q, 4096x2560 o, 2560x1024 k/v).
// All timings are GPU-side (command buffer GPUStartTime..GPUEndTime), median of N runs.
//
//   swiftc -O gemv_bench.swift -o gemv_bench && ./gemv_bench [iters]
import Metal
import Foundation

let src = """
#include <metal_stdlib>
using namespace metal;

// ---- [A] two-pass step 1: int4 (16 nibbles per uint2) -> fp16, fp16-arithmetic unpack via
//      half(byte)*0.0625 / floor / frac*16, scale + zero applied per group of 32 along K.
//      Layout: packed[K/16 * N] uint2, row-major over K for each output column n
//      (weights stored as N rows of K nibbles). One thread = 16 consecutive K values.
kernel void dequant_int4_to_half(device const uint2* packed [[buffer(0)]],
                                 device const half* scales [[buffer(1)]],   // N * (K/32)
                                 device const half* zeros  [[buffer(2)]],   // N * (K/32)
                                 device half4* out [[buffer(3)]],           // N * K halves
                                 constant uint& K [[buffer(4)]],
                                 constant uint& N [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
  uint per_row = K / 16;
  if (gid >= per_row * N) return;
  uint n = gid / per_row, k16 = gid % per_row;
  uint2 v = packed[gid];
  half4 a, b, c, d;
  half4 lo = half4(half(v.x & 255u), half((v.x >> 8u) & 255u), half((v.x >> 16u) & 255u), half((v.x >> 24u) & 255u)) * 0.0625h;
  half4 hi = half4(half(v.y & 255u), half((v.y >> 8u) & 255u), half((v.y >> 16u) & 255u), half((v.y >> 24u) & 255u)) * 0.0625h;
  a.y = floor(lo.x); a.w = floor(lo.y); a.x = (lo.x - a.y) * 16.0h; a.z = (lo.y - a.w) * 16.0h;
  b.y = floor(lo.z); b.w = floor(lo.w); b.x = (lo.z - b.y) * 16.0h; b.z = (lo.w - b.w) * 16.0h;
  c.y = floor(hi.x); c.w = floor(hi.y); c.x = (hi.x - c.y) * 16.0h; c.z = (hi.y - c.w) * 16.0h;
  d.y = floor(hi.z); d.w = floor(hi.w); d.x = (hi.z - d.y) * 16.0h; d.z = (hi.w - d.w) * 16.0h;
  uint g = n * (K / 32) + (k16 * 16) / 32;
  half s = scales[g]; half z = -s * (8.0h + zeros[g]);
  uint o = (n * K + k16 * 16) / 4;
  out[o] = a * s + z; out[o + 1] = b * s + z; out[o + 2] = c * s + z; out[o + 3] = d * s + z;
}

// ---- [B] two-pass step 2: fp16 GEMV y[n] = sum_k W[n][k] * x[k]; one simdgroup per row.
kernel void gemv_half(device const half4* W [[buffer(0)]],   // N * K/4
                      device const half4* x [[buffer(1)]],   // K/4
                      device half* y [[buffer(2)]],
                      constant uint& K [[buffer(3)]],
                      constant uint& N [[buffer(4)]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint tgid [[threadgroup_position_in_grid]],
                      uint simd_lane [[thread_index_in_simdgroup]],
                      uint simd_id [[simdgroup_index_in_threadgroup]]) {
  uint n = tgid * 4 + simd_id;              // 128 threads = 4 simdgroups = 4 rows
  if (n >= N) return;
  uint k4 = K / 4;
  float acc = 0.0f;
  for (uint i = simd_lane; i < k4; i += 32) acc += float(dot(W[n * k4 + i], x[i]));
  acc = simd_sum(acc);
  if (simd_lane == 0) y[n] = half(acc);
}

// ---- [C] fused: int4 read + in-register dequant + dot, one simdgroup per row, 32 lanes each
//      own 16 consecutive K values (one uint2) per step -> a full gs32 group is 2 lanes.
kernel void gemv_int4_fused(device const uint2* packed [[buffer(0)]],
                            device const half* scales [[buffer(1)]],
                            device const half* zeros [[buffer(2)]],
                            device const half4* x [[buffer(3)]],
                            device half* y [[buffer(4)]],
                            constant uint& K [[buffer(5)]],
                            constant uint& N [[buffer(6)]],
                            uint tgid [[threadgroup_position_in_grid]],
                            uint simd_lane [[thread_index_in_simdgroup]],
                            uint simd_id [[simdgroup_index_in_threadgroup]]) {
  uint n = tgid * 4 + simd_id;
  if (n >= N) return;
  uint per_row = K / 16;
  float acc = 0.0f;
  for (uint k16 = simd_lane; k16 < per_row; k16 += 32) {
    uint2 v = packed[n * per_row + k16];
    uint g = n * (K / 32) + k16 / 2;
    half s = scales[g]; half z = -s * (8.0h + zeros[g]);
    // nibble unpack with integer ops (low nibble first), 16 values
    half4 w0 = half4(half(v.x & 15u), half((v.x >> 4u) & 15u), half((v.x >> 8u) & 15u), half((v.x >> 12u) & 15u));
    half4 w1 = half4(half((v.x >> 16u) & 15u), half((v.x >> 20u) & 15u), half((v.x >> 24u) & 15u), half(v.x >> 28u));
    half4 w2 = half4(half(v.y & 15u), half((v.y >> 4u) & 15u), half((v.y >> 8u) & 15u), half((v.y >> 12u) & 15u));
    half4 w3 = half4(half((v.y >> 16u) & 15u), half((v.y >> 20u) & 15u), half((v.y >> 24u) & 15u), half(v.y >> 28u));
    uint xo = k16 * 4;
    half4 xs = x[xo] + x[xo + 1] + x[xo + 2] + x[xo + 3];
    // (w*s + z) . x  ==  s * (w . x) + z * sum(x)
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
let pDq = pso("dequant_int4_to_half"), pGh = pso("gemv_half"), pF = pso("gemv_int4_fused")
let iters = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1])! : 40

struct Shape { let name: String; let K: Int; let N: Int; let perToken: Int }
// Qwen3-4B decode matrices (K = input dim, N = output rows) and how many times each shape is
// streamed per token (36 layers; gate+up and k+v share a shape; lm_head once)
let shapes = [Shape(name: "gate/up 2560x9728", K: 2560, N: 9728, perToken: 72), Shape(name: "down 9728x2560", K: 9728, N: 2560, perToken: 36),
              Shape(name: "q 2560x4096", K: 2560, N: 4096, perToken: 36), Shape(name: "o 4096x2560", K: 4096, N: 2560, perToken: 36),
              Shape(name: "k/v 2560x1024", K: 2560, N: 1024, perToken: 72), Shape(name: "lm_head 2560x151936", K: 2560, N: 151936, perToken: 1)]

let usePrivate = CommandLine.arguments.contains("--private")
func buf(_ bytes: Int) -> MTLBuffer { dev.makeBuffer(length: bytes, options: .storageModeShared)! }
// GPU-only buffer initialised from a shared staging copy (delegates allocate weights and
// scratch as private/GPU-only memory; the storage mode changes the cache policy)
func toPrivate(_ src: MTLBuffer) -> MTLBuffer {
  if !usePrivate { return src }
  let d = dev.makeBuffer(length: src.length, options: .storageModePrivate)!
  let cb = q.makeCommandBuffer()!; let b = cb.makeBlitCommandEncoder()!; b.copy(from: src, sourceOffset: 0, to: d, destinationOffset: 0, size: src.length); b.endEncoding(); cb.commit(); cb.waitUntilCompleted(); return d
}
func scratchBuf(_ bytes: Int) -> MTLBuffer { dev.makeBuffer(length: bytes, options: usePrivate ? .storageModePrivate : .storageModeShared)! }
func fill(_ b: MTLBuffer, _ seed: UInt32) { let p = b.contents().assumingMemoryBound(to: UInt32.self); var s = seed &+ 1; for i in 0..<(b.length / 4) { s = s &* 1664525 &+ 1013904223; p[i] = s } }
func fillHalf(_ b: MTLBuffer, _ v: Float) { let p = b.contents().assumingMemoryBound(to: UInt16.self); let h = Float16(v); let bits = h.bitPattern; for i in 0..<(b.length / 2) { p[i] = bits } }

func timeGPU(_ enc: (MTLComputeCommandEncoder) -> Void) -> Double {
  let cb = q.makeCommandBuffer()!; let e = cb.makeComputeCommandEncoder()!; enc(e); e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
  return cb.gpuEndTime - cb.gpuStartTime
}
func median(_ a: [Double]) -> Double { let s = a.sorted(); return s[s.count / 2] }
func pad(_ s: String, _ n: Int) -> String { s.count >= n ? s : s + String(repeating: " ", count: n - s.count) }

print("device \(dev.name)  iters \(iters)  storage \(usePrivate ? "private (GPU-only)" : "shared")  (GPU-side ms per matrix, median; GB/s = int4 payload bytes incl. scales/zeros ÷ time)")
print(pad("matrix", 21) + " A dequant B fp16gemv        A+B    C fused |  A+B GB/s    C GB/s fp16 GB/s   (ms)")
var tokAB = 0.0, tokC = 0.0, tokB = 0.0, tokA = 0.0
var checked = false
for sh in shapes {
  let K = sh.K, N = sh.N, groups = K / 32
  let packed = buf(K / 2 * N), scales = buf(N * groups * 2), zeros = buf(N * groups * 2)
  let wHalf = buf(K * N * 2), x = buf(K * 2), y = buf(N * 2)
  fill(packed, 7); fillHalf(scales, 0.01); fillHalf(zeros, 7.5); fillHalf(x, 0.5)
  var k = UInt32(K), n = UInt32(N)
  var tA: [Double] = [], tB: [Double] = [], tC: [Double] = []
  for _ in 0..<(iters + 5) {
    tA.append(timeGPU { e in
      e.setComputePipelineState(pDq); e.setBuffer(packed, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1); e.setBuffer(zeros, offset: 0, index: 2); e.setBuffer(wHalf, offset: 0, index: 3)
      e.setBytes(&k, length: 4, index: 4); e.setBytes(&n, length: 4, index: 5)
      let threads = K / 16 * N; e.dispatchThreads(MTLSize(width: threads, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)) })
    tB.append(timeGPU { e in
      e.setComputePipelineState(pGh); e.setBuffer(wHalf, offset: 0, index: 0); e.setBuffer(x, offset: 0, index: 1); e.setBuffer(y, offset: 0, index: 2)
      e.setBytes(&k, length: 4, index: 3); e.setBytes(&n, length: 4, index: 4)
      e.dispatchThreadgroups(MTLSize(width: (N + 3) / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)) })
    tC.append(timeGPU { e in
      e.setComputePipelineState(pF); e.setBuffer(packed, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1); e.setBuffer(zeros, offset: 0, index: 2); e.setBuffer(x, offset: 0, index: 3); e.setBuffer(y, offset: 0, index: 4)
      e.setBytes(&k, length: 4, index: 5); e.setBytes(&n, length: 4, index: 6)
      e.dispatchThreadgroups(MTLSize(width: (N + 3) / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)) })
  }
  tA.removeFirst(5); tB.removeFirst(5); tC.removeFirst(5)
  let a = median(tA), b = median(tB), c = median(tC)
  let payload = Double(K / 2 * N + 2 * N * groups * 2)   // int4 nibbles + fp16 scales + fp16 zeros
  let fp16bytes = Double(K * N * 2)
  tokA += a * Double(sh.perToken); tokB += b * Double(sh.perToken); tokAB += (a + b) * Double(sh.perToken); tokC += c * Double(sh.perToken)
  print(pad(sh.name, 21) + String(format: "%10.3f %10.3f %10.3f %10.3f | %9.0f %9.0f %9.0f", a * 1e3, b * 1e3, (a + b) * 1e3, c * 1e3, payload / (a + b) / 1e9, payload / c / 1e9, fp16bytes / b / 1e9))
  if !checked {
    // correctness: the fused kernel must reproduce two-pass (A then B) within fp16 tolerance.
    // y currently holds C's result (last run); recompute A+B into yRef.
    checked = true
    let yC = Array(UnsafeBufferPointer(start: y.contents().assumingMemoryBound(to: Float16.self), count: N))
    _ = timeGPU { e in
      e.setComputePipelineState(pDq); e.setBuffer(packed, offset: 0, index: 0); e.setBuffer(scales, offset: 0, index: 1); e.setBuffer(zeros, offset: 0, index: 2); e.setBuffer(wHalf, offset: 0, index: 3)
      e.setBytes(&k, length: 4, index: 4); e.setBytes(&n, length: 4, index: 5)
      e.dispatchThreads(MTLSize(width: K / 16 * N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)) }
    _ = timeGPU { e in
      e.setComputePipelineState(pGh); e.setBuffer(wHalf, offset: 0, index: 0); e.setBuffer(x, offset: 0, index: 1); e.setBuffer(y, offset: 0, index: 2)
      e.setBytes(&k, length: 4, index: 3); e.setBytes(&n, length: 4, index: 4)
      e.dispatchThreadgroups(MTLSize(width: (N + 3) / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)) }
    let yAB = Array(UnsafeBufferPointer(start: y.contents().assumingMemoryBound(to: Float16.self), count: N))
    var maxRel = 0.0, maxAbs = 0.0
    for i in 0..<N { let d = abs(Double(yC[i]) - Double(yAB[i])); maxAbs = max(maxAbs, d); maxRel = max(maxRel, d / max(abs(Double(yAB[i])), 1e-3)) }
    print("  correctness (" + sh.name + "): " + String(format: "fused vs two-pass  max|diff| %.4g  max rel %.4g  (y[0] %.4f vs %.4f)", maxAbs, maxRel, Double(yC[0]), Double(yAB[0])))
  }
}
print(String(format: "per-token model from per-dispatch timings (36 layers x 7 matrices + lm_head): A dequant %.2f ms + B fp16 GEMV %.2f ms = two-pass %.2f ms   vs   fused C %.2f ms   (%.0f%% of two-pass)", tokA * 1e3, tokB * 1e3, tokAB * 1e3, tokC * 1e3, tokC / tokAB * 100))

// ---- whole-token command buffers: every matrix of every layer back-to-back in ONE command
//      buffer (the way the delegate encodes a token: no CPU sync between dispatches), so
//      per-dispatch launch overhead is not counted 253 times. Distinct buffers per layer so
//      nothing stays cache-resident across layers (36 x 50 MB int4 = 1.8 GB streamed).
struct Mat { let K: Int; let N: Int; let packed: MTLBuffer; let scales: MTLBuffer; let zeros: MTLBuffer; var wHalf: MTLBuffer; let y: MTLBuffer }
// fp16 scratch for the two-pass path: `shared` = one scratch buffer per matrix shape, reused
// by every layer (what a delegate does; the buffer can stay cache-resident between the
// dequant write and the GEMV read); `fresh` = a distinct buffer per matrix (pessimistic).
var scratch: [String: MTLBuffer] = [:]
func mkMat(_ K: Int, _ N: Int, _ seed: UInt32, shared: Bool) -> Mat {
  let key = "\(K)x\(N)"
  if shared && scratch[key] == nil { scratch[key] = scratchBuf(K * N * 2) }
  let pk = buf(K / 2 * N), sc = buf(N * (K / 32) * 2), zr = buf(N * (K / 32) * 2)
  fill(pk, seed); fillHalf(sc, 0.01); fillHalf(zr, 7.5)
  return Mat(K: K, N: N, packed: toPrivate(pk), scales: toPrivate(sc), zeros: toPrivate(zr), wHalf: shared ? scratch[key]! : scratchBuf(K * N * 2), y: buf(N * 2))
}
let layerShapes = [(2560, 9728), (2560, 9728), (9728, 2560), (2560, 4096), (4096, 2560), (2560, 1024), (2560, 1024)]
var layers: [[Mat]] = []
for l in 0..<36 { layers.append(layerShapes.enumerated().map { mkMat($0.element.0, $0.element.1, UInt32(l * 7 + $0.offset), shared: true) }) }
var head = mkMat(2560, 151936, 999, shared: true)
let xs = buf(9728 * 2); fillHalf(xs, 0.5)
func encTwoPass(_ e: MTLComputeCommandEncoder, _ m: Mat) {
  var k = UInt32(m.K), n = UInt32(m.N)
  e.setComputePipelineState(pDq); e.setBuffer(m.packed, offset: 0, index: 0); e.setBuffer(m.scales, offset: 0, index: 1); e.setBuffer(m.zeros, offset: 0, index: 2); e.setBuffer(m.wHalf, offset: 0, index: 3)
  e.setBytes(&k, length: 4, index: 4); e.setBytes(&n, length: 4, index: 5)
  e.dispatchThreads(MTLSize(width: m.K / 16 * m.N, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
  e.setComputePipelineState(pGh); e.setBuffer(m.wHalf, offset: 0, index: 0); e.setBuffer(xs, offset: 0, index: 1); e.setBuffer(m.y, offset: 0, index: 2)
  e.setBytes(&k, length: 4, index: 3); e.setBytes(&n, length: 4, index: 4)
  e.dispatchThreadgroups(MTLSize(width: (m.N + 3) / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
}
func encFused(_ e: MTLComputeCommandEncoder, _ m: Mat) {
  var k = UInt32(m.K), n = UInt32(m.N)
  e.setComputePipelineState(pF); e.setBuffer(m.packed, offset: 0, index: 0); e.setBuffer(m.scales, offset: 0, index: 1); e.setBuffer(m.zeros, offset: 0, index: 2); e.setBuffer(xs, offset: 0, index: 3); e.setBuffer(m.y, offset: 0, index: 4)
  e.setBytes(&k, length: 4, index: 5); e.setBytes(&n, length: 4, index: 6)
  e.dispatchThreadgroups(MTLSize(width: (m.N + 3) / 4, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
}
func tokenTime(fused: Bool, withHead: Bool) -> Double {
  var ts: [Double] = []
  for _ in 0..<(iters / 2 + 3) {
    ts.append(timeGPU { e in
      for layer in layers { for m in layer { fused ? encFused(e, m) : encTwoPass(e, m) } }
      if withHead { fused ? encFused(e, head) : encTwoPass(e, head) }
    })
  }
  ts.removeFirst(3); return median(ts)
}
let int4Bytes = Double(layers.flatMap { $0 }.reduce(0) { $0 + $1.K / 2 * $1.N + 2 * $1.N * ($1.K / 32) * 2 } + head.K / 2 * head.N + 2 * head.N * (head.K / 32) * 2)
let tp = tokenTime(fused: false, withHead: true), fu = tokenTime(fused: true, withHead: true)
let tpL = tokenTime(fused: false, withHead: false), fuL = tokenTime(fused: true, withHead: false)
print(String(format: "whole-token command buffer (36 layers x 7 + lm_head, %.2f GB int4 payload), fp16 scratch shared per shape:", int4Bytes / 1e9))
print(String(format: "  two-pass %.2f ms (%.0f GB/s of int4 payload)   fused %.2f ms (%.0f GB/s)   -> fused = %.0f%% of two-pass", tp * 1e3, int4Bytes / tp / 1e9, fu * 1e3, int4Bytes / fu / 1e9, fu / tp * 100))
print(String(format: "  layers only (no lm_head): two-pass %.2f ms   fused %.2f ms;   lm_head alone: two-pass %.2f ms   fused %.2f ms", tpL * 1e3, fuL * 1e3, (tp - tpL) * 1e3, (fu - fuL) * 1e3))
// pessimistic variant: a fresh fp16 scratch per matrix (7.3 GB of intermediates per token)
for l in 0..<36 { for i in 0..<7 { layers[l][i].wHalf = scratchBuf(layers[l][i].K * layers[l][i].N * 2) } }
head.wHalf = scratchBuf(head.K * head.N * 2)
let tpF = tokenTime(fused: false, withHead: true)
print(String(format: "  two-pass with a fresh fp16 scratch per matrix (no cache reuse possible): %.2f ms", tpF * 1e3))
