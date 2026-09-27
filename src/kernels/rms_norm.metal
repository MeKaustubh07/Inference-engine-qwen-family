#include <metal_stdlib>
using namespace metal;

// RMSNorm: one threadgroup per row (token). Threads stride over the row, reduce the sum of squares
// inside each 32-wide SIMD group with simd_sum, combine the SIMD-group partials through threadgroup memory,
// then every thread scales its share of the row.
kernel void rms_norm(device float* out        [[buffer(0)]],
                     device const float* x    [[buffer(1)]],
                     device const bfloat* w   [[buffer(2)]],
                     constant float& eps      [[buffer(3)]],
                     constant uint& d         [[buffer(4)]],
                     uint row  [[threadgroup_position_in_grid]],
                     uint tid  [[thread_position_in_threadgroup]],
                     uint ntg  [[threads_per_threadgroup]],
                     uint sg   [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    device const float* xr = x + row * d;
    float acc = 0.0f;
    for (uint j = tid; j < d; j += ntg) acc += xr[j] * xr[j];
    acc = simd_sum(acc);
    if (lane == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = lane < (ntg + 31) / 32 ? partial[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) partial[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = precise::rsqrt(partial[0] / float(d) + eps);
    for (uint j = tid; j < d; j += ntg) out[row * d + j] = xr[j] * inv * float(w[j]);
}
