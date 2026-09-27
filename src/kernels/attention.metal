#include <metal_stdlib>
using namespace metal;

// Decode attention: ONE new query per head against all S cached keys/values (GQA-aware).
// One threadgroup per query head.  q: [Hq, d]  k, v: [S, Hkv, d]  scores (scratch): [Hq, S]  out: [Hq, d]
// Steps: scores = q.k / sqrt(d) -> max -> exp and sum (numerically stable softmax) -> weighted sum of v.
// No mask is needed: the single query is the newest position, so every cached key is in its past.
kernel void attention_decode(device float* out        [[buffer(0)]],
                             device const float* q    [[buffer(1)]],
                             device const float* k    [[buffer(2)]],
                             device const float* v    [[buffer(3)]],
                             device float* scores     [[buffer(4)]],
                             constant uint& S         [[buffer(5)]],
                             constant uint& n_kv      [[buffer(6)]],
                             constant uint& group     [[buffer(7)]],
                             constant uint& d         [[buffer(8)]],
                             uint h    [[threadgroup_position_in_grid]],
                             uint tid  [[thread_position_in_threadgroup]],
                             uint ntg  [[threads_per_threadgroup]],
                             uint sg   [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[32];
    uint kvh = h / group;
    device const float* qh = q + h * d;
    device float* sc = scores + (ulong)h * S;
    float scale = precise::rsqrt(float(d));

    // 1. scores and running max
    float m = -INFINITY;
    for (uint s = tid; s < S; s += ntg) {
        device const float4* ks = (device const float4*)(k + ((ulong)s * n_kv + kvh) * d);
        device const float4* q4 = (device const float4*)qh;
        float dotv = 0.0f;
        for (uint j = 0; j < d / 4; ++j) dotv += dot(q4[j], ks[j]);     // d is a multiple of 4 (64 or 256)
        dotv *= scale;
        sc[s] = dotv;
        m = max(m, dotv);
    }
    m = simd_max(m);
    if (lane == 0) red[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) { float t = lane < (ntg + 31) / 32 ? red[lane] : -INFINITY; t = simd_max(t); if (lane == 0) red[0] = t; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. exponentials and their sum
    float sum = 0.0f;
    for (uint s = tid; s < S; s += ntg) { float e = precise::exp(sc[s] - m); sc[s] = e; sum += e; }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (sg == 0) { float t = lane < (ntg + 31) / 32 ? red[lane] : 0.0f; t = simd_sum(t); if (lane == 0) red[0] = t; }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    float inv = 1.0f / red[0];

    // 3. weighted sum of values. The threadgroup is split into ntg/d groups; group p handles positions
    //    p, p + ngroups, ... for one output dimension each, then the partial sums are added.
    threadgroup float part[1024];
    uint ngroups = max(1u, ntg / d);
    uint j = tid % d, p = tid / d;
    if (p < ngroups) {
        float acc = 0.0f;
        for (uint s = p; s < S; s += ngroups) acc += sc[s] * v[((ulong)s * n_kv + kvh) * d + j];
        part[p * d + j] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p == 0) {
        float acc = 0.0f;
        for (uint g = 0; g < ngroups; ++g) acc += part[g * d + j];
        out[h * d + j] = acc * inv;
    }
}
