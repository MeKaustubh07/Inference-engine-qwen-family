#include <metal_stdlib>
using namespace metal;

// Qwen3.5 Gated DeltaNet, single-token decode. Two fused kernels replace ~20 small tensor ops per layer.

// Causal depthwise conv (kernel 4) + SiLU for one new token, updating the 3-token history in place.
// x: raw in_proj_qkv output [C]; tail: [n_linear, 3, C] (this layer's slot); w: [C, 4] (tap 3 = current token).
kernel void conv_step(device float* u          [[buffer(0)]],
                      device const float* x    [[buffer(1)]],
                      device float* tail       [[buffer(2)]],
                      device const float* w    [[buffer(3)]],
                      constant uint& C         [[buffer(4)]],
                      constant uint& slot      [[buffer(5)]],
                      uint c [[thread_position_in_grid]]) {
    if (c >= C) return;
    device float* t = tail + (ulong)slot * 3 * C;
    float h0 = t[c], h1 = t[C + c], h2 = t[2 * C + c], xc = x[c];
    device const float* wc = w + (ulong)c * 4;
    float y = wc[0] * h0 + wc[1] * h1 + wc[2] * h2 + wc[3] * xc;
    u[c] = y / (1.0f + precise::exp(-y));
    t[c] = h1; t[C + c] = h2; t[2 * C + c] = xc;               // shift the window: keep the last 3 raw inputs
}

// One threadgroup per head, one thread per value column j (dv threads; dk == dv == 128 here).
// l2norm(q), l2norm(k), q *= 1/sqrt(dk); beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
// S = exp(g) * S;  delta = beta * (v - k^T S);  S += k delta^T;  o = q^T S;  out = rmsnorm(o) * w * silu(z)
kernel void gdn_decode(device float* out             [[buffer(0)]],    // [H * dv]
                       device const float* u         [[buffer(1)]],    // conv output [q(H*dk) | k(H*dk) | v(H*dv)]
                       device const float* z         [[buffer(2)]],    // [H * dv]
                       device const float* bl        [[buffer(3)]],    // beta logits [H]
                       device const float* a         [[buffer(4)]],    // decay input [H]
                       device const float* A_log     [[buffer(5)]],    // [H]
                       device const float* dt_bias   [[buffer(6)]],    // [H]
                       device const float* norm_w    [[buffer(7)]],    // [dv], plain w
                       device float* S               [[buffer(8)]],    // [n_linear, H, dk, dv] fp32, updated in place
                       constant uint& H              [[buffer(9)]],
                       constant uint& dk             [[buffer(10)]],
                       constant uint& slot           [[buffer(11)]],
                       constant float& eps           [[buffer(12)]],
                       uint h    [[threadgroup_position_in_grid]],
                       uint j    [[thread_position_in_threadgroup]],
                       uint ntg  [[threads_per_threadgroup]],
                       uint sg   [[simdgroup_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[128], ks[128], red[3][4];
    uint dv = ntg, nsg = (ntg + 31) / 32;
    float qj = u[h * dk + j], kj = u[H * dk + h * dk + j], vj = u[2 * H * dk + h * dv + j];

    // l2 norms of q and k over the head (sum of squares, eps 1e-6), then q scale
    float sq = simd_sum(qj * qj), sk = simd_sum(kj * kj);
    if (lane == 0) { red[0][sg] = sq; red[1][sg] = sk; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sq = 0.0f; sk = 0.0f;
    for (uint i = 0; i < nsg; ++i) { sq += red[0][i]; sk += red[1][i]; }
    qs[j] = qj * precise::rsqrt(sq + 1e-6f) * precise::rsqrt(float(dk));
    ks[j] = kj * precise::rsqrt(sk + 1e-6f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float beta = 1.0f / (1.0f + precise::exp(-bl[h]));
    float ab = a[h] + dt_bias[h];
    float sp = ab > 20.0f ? ab : precise::log(1.0f + precise::exp(ab));
    float decay = precise::exp(-precise::exp(A_log[h]) * sp);

    device float* Sh = S + ((ulong)slot * H + h) * dk * dv;     // this head's [dk, dv] state, column j
    float kv = 0.0f;
    for (uint i = 0; i < dk; ++i) kv += decay * Sh[i * dv + j] * ks[i];
    float delta = beta * (vj - kv);
    float o = 0.0f;
    for (uint i = 0; i < dk; ++i) {
        float s = decay * Sh[i * dv + j] + ks[i] * delta;
        Sh[i * dv + j] = s;
        o += s * qs[i];                                          // output reads the UPDATED state
    }

    // gated RMSNorm over the head's dv outputs: rmsnorm(o) * w * silu(z)
    float so = simd_sum(o * o);
    if (lane == 0) red[2][sg] = so;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    so = 0.0f;
    for (uint i = 0; i < nsg; ++i) so += red[2][i];
    float zj = z[h * dv + j];
    out[h * dv + j] = o * precise::rsqrt(so / float(dv) + eps) * norm_w[j] * (zj / (1.0f + precise::exp(-zj)));
}
