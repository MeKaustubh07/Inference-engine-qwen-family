#include <metal_stdlib>
using namespace metal;

// Prefill linear layer: Y[T, N] = X[T, K] (fp32) . W[N, K]^T (bf16), fp32 accumulation.
// Classic shared-memory tiling: each 16x16 threadgroup computes a 16x16 tile of Y. For every step along K
// it cooperatively loads a 16x16 tile of X and of W into threadgroup memory (one element per thread), then
// every thread reuses those 32 loaded values 16 times. Each weight read from device memory is reused 16x.
constant constexpr uint TILE = 16;

kernel void gemm_bf16(device float* Y          [[buffer(0)]],
                      device const float* X    [[buffer(1)]],
                      device const bfloat* W   [[buffer(2)]],
                      constant uint& T         [[buffer(3)]],
                      constant uint& N         [[buffer(4)]],
                      constant uint& K         [[buffer(5)]],
                      uint2 tg  [[threadgroup_position_in_grid]],
                      uint2 tid [[thread_position_in_threadgroup]]) {
    threadgroup float xs[TILE][TILE];
    threadgroup float ws[TILE][TILE];
    uint row = tg.y * TILE + tid.y;           // token index in Y
    uint col = tg.x * TILE + tid.x;           // output feature in Y
    float acc = 0.0f;
    for (uint k0 = 0; k0 < K; k0 += TILE) {
        uint kx = k0 + tid.x, kw = k0 + tid.y;
        xs[tid.y][tid.x] = (row < T && kx < K) ? X[(ulong)row * K + kx] : 0.0f;
        uint wrow = tg.x * TILE + tid.x;      // W row feeding output column `col`
        ws[tid.y][tid.x] = (wrow < N && kw < K) ? float(W[(ulong)wrow * K + kw]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < TILE; ++k) acc += xs[tid.y][k] * ws[k][tid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < T && col < N) Y[(ulong)row * N + col] = acc;
}
