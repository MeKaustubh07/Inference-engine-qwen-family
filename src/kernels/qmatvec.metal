#include <metal_stdlib>
using namespace metal;

// Quantized decode matvec: y[N] = dequant(W)[N, K] . x[K] (+ bias) (+ residual), fp32 accumulation.
// Weights are dequantized in registers; the full-precision matrix never exists in memory.
// One 32-thread SIMD group per output row; scales are fp16, one per 32 weights (BLOCK) along the row.

kernel void matvec_q8(device float* y           [[buffer(0)]],
                      device const char4* W     [[buffer(1)]],   // int8 [N, K] read 4 at a time
                      device const half* scales [[buffer(2)]],   // [N, K/32]
                      device const float4* x    [[buffer(3)]],
                      device const bfloat* b    [[buffer(4)]],
                      device const float* res   [[buffer(5)]],
                      constant uint& K          [[buffer(6)]],
                      constant uint& N          [[buffer(7)]],
                      constant uint& has_bias   [[buffer(8)]],
                      constant uint& has_res    [[buffer(9)]],
                      uint gid  [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    if (row >= N) return;
    device const char4* wr = W + (ulong)row * (K / 4);
    device const half* sr = scales + (ulong)row * (K / 32);
    float acc = 0.0f;
    for (uint j = lane; j < K / 4; j += 32) acc += dot(float4(wr[j]), x[j]) * float(sr[j / 8]);   // 8 char4 per block
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc + (has_bias ? float(b[row]) : 0.0f) + (has_res ? res[row] : 0.0f);
}

kernel void matvec_q4(device float* y           [[buffer(0)]],
                      device const uchar4* W    [[buffer(1)]],   // packed [N, K/2]: 4 bytes = 8 weights
                      device const half* scales [[buffer(2)]],
                      device const float4* x    [[buffer(3)]],
                      device const bfloat* b    [[buffer(4)]],
                      device const float* res   [[buffer(5)]],
                      constant uint& K          [[buffer(6)]],
                      constant uint& N          [[buffer(7)]],
                      constant uint& has_bias   [[buffer(8)]],
                      constant uint& has_res    [[buffer(9)]],
                      device const half* mins   [[buffer(10)]],  // asymmetric: w = q * scale + min, q in [0, 15]
                      uint gid  [[thread_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row = gid / 32;
    if (row >= N) return;
    device const uchar4* wr = W + (ulong)row * (K / 8);
    device const half* sr = scales + (ulong)row * (K / 32);
    device const half* mr = mins + (ulong)row * (K / 32);
    float acc = 0.0f;
    for (uint j = lane; j < K / 8; j += 32) {                     // 8 weights per step, 4 steps per block
        uchar4 p = wr[j];
        float4 lo = float4(p & 0x0F);                             // elements 0,2,4,6 of the 8
        float4 hi = float4(p >> 4);                               // elements 1,3,5,7
        float4 xa = x[2 * j], xb = x[2 * j + 1];                  // x[8j .. 8j+7]
        float qdot = dot(float4(lo.x, hi.x, lo.y, hi.y), xa) + dot(float4(lo.z, hi.z, lo.w, hi.w), xb);
        float xsum = xa.x + xa.y + xa.z + xa.w + xb.x + xb.y + xb.z + xb.w;
        acc += qdot * float(sr[j / 4]) + xsum * float(mr[j / 4]); // sum((q*s + m) * x) = s*q.x + m*sum(x)
    }
    acc = simd_sum(acc);
    if (lane == 0) y[row] = acc + (has_bias ? float(b[row]) : 0.0f) + (has_res ? res[row] : 0.0f);
}
