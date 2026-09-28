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

// Batched decode (continuous batching), M <= MAX_BATCH sequences share one pass over the weights: each weight
// block is read and dequantized once for all M activation rows, 2 output rows per SIMD group (see matvec.metal
// for RR, MAX_BATCH and ROWS_EPILOGUE). x and y are row-major [M, K] and [M, N].
kernel void matvec_q4_rows(device float* y [[buffer(0)]], device const uchar4* W [[buffer(1)]],
                           device const half* scales [[buffer(2)]], device const float4* x [[buffer(3)]],
                           device const bfloat* b [[buffer(4)]], device const float* res [[buffer(5)]],
                           constant uint& K [[buffer(6)]], constant uint& N [[buffer(7)]],
                           constant uint& has_bias [[buffer(8)]], constant uint& has_res [[buffer(9)]],
                           device const half* mins [[buffer(10)]], constant uint& M [[buffer(11)]],
                           uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * RR;
    if (row0 >= N) return;
    float acc[RR][MAX_BATCH];
    for (uint r = 0; r < RR; r++) for (uint m = 0; m < MAX_BATCH; m++) acc[r][m] = 0.0f;
    for (uint j = lane; j < K / 8; j += 32) {
        float4 wa[RR], wb[RR];
        for (uint r = 0; r < RR; r++) {
            ulong rr = min(row0 + r, N - 1);
            uchar4 p = W[rr * (K / 8) + j];
            float s = float(scales[rr * (K / 32) + j / 4]), mn = float(mins[rr * (K / 32) + j / 4]);
            float4 lo = float4(p & 0x0F), hi = float4(p >> 4);
            wa[r] = float4(lo.x, hi.x, lo.y, hi.y) * s + mn;
            wb[r] = float4(lo.z, hi.z, lo.w, hi.w) * s + mn;
        }
        for (uint m = 0; m < MAX_BATCH; m++) {
            if (m < M) {
                float4 xa = x[m * (K / 4) + 2 * j], xb = x[m * (K / 4) + 2 * j + 1];
                for (uint r = 0; r < RR; r++) acc[r][m] += dot(wa[r], xa) + dot(wb[r], xb);
            }
        }
    }
    ROWS_EPILOGUE
}

kernel void matvec_q8_rows(device float* y [[buffer(0)]], device const char4* W [[buffer(1)]],
                           device const half* scales [[buffer(2)]], device const float4* x [[buffer(3)]],
                           device const bfloat* b [[buffer(4)]], device const float* res [[buffer(5)]],
                           constant uint& K [[buffer(6)]], constant uint& N [[buffer(7)]],
                           constant uint& has_bias [[buffer(8)]], constant uint& has_res [[buffer(9)]],
                           constant uint& M [[buffer(10)]],
                           uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (gid / 32) * RR;
    if (row0 >= N) return;
    float acc[RR][MAX_BATCH];
    for (uint r = 0; r < RR; r++) for (uint m = 0; m < MAX_BATCH; m++) acc[r][m] = 0.0f;
    for (uint j = lane; j < K / 4; j += 32) {
        float4 w[RR];
        for (uint r = 0; r < RR; r++) {
            ulong rr = min(row0 + r, N - 1);
            w[r] = float4(W[rr * (K / 4) + j]) * float(scales[rr * (K / 32) + j / 8]);
        }
        for (uint m = 0; m < MAX_BATCH; m++) {
            if (m < M) {
                float4 xv = x[m * (K / 4) + j];
                for (uint r = 0; r < RR; r++) acc[r][m] += dot(w[r], xv);
            }
        }
    }
    ROWS_EPILOGUE
}

// Prefill: expand quantized weights to fp32 in one pass (read packed blocks, write fp32), feeding the tuned GEMM.
// fp32, not bf16: on the M2, MPS runs fp32 GEMM at ~2.5 TFLOPS and bf16 at ~1.4 (no native bf16 math before M3).
// One thread per 8 (INT4) or 4 (INT8) weights; W/scales/mins are the rows of one chunk, row-major.
kernel void dequant_q4_f32(device float4* out           [[buffer(0)]],
                           device const uchar4* W       [[buffer(1)]],
                           device const half* scales    [[buffer(2)]],
                           device const half* mins      [[buffer(3)]],
                           constant uint& K             [[buffer(4)]],
                           constant uint& n             [[buffer(5)]],    // uchar4 groups in the chunk
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint row = gid / (K / 8), j = gid - row * (K / 8);
    uchar4 p = W[gid];
    float s = float(scales[row * (K / 32) + j / 4]), mn = float(mins[row * (K / 32) + j / 4]);
    float4 lo = float4(p & 0x0F) * s + mn, hi = float4(p >> 4) * s + mn;   // low nibble = even element
    out[2 * gid] = float4(lo.x, hi.x, lo.y, hi.y);
    out[2 * gid + 1] = float4(lo.z, hi.z, lo.w, hi.w);
}

kernel void dequant_q8_f32(device float4* out           [[buffer(0)]],
                           device const char4* W        [[buffer(1)]],
                           device const half* scales    [[buffer(2)]],
                           constant uint& K             [[buffer(3)]],
                           constant uint& n             [[buffer(4)]],    // char4 groups in the chunk
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint row = gid / (K / 4), j = gid - row * (K / 4);
    out[gid] = float4(W[gid]) * float(scales[row * (K / 32) + j / 8]);
}
