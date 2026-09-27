#include <metal_stdlib>
using namespace metal;

// RoPE, Qwen2 "rotate_half" pairing: number i pairs with number i + d/2.
// One thread per (token, head, pair). x/out: [T, H, d] contiguous, positions: [T].
kernel void rope(device float* out           [[buffer(0)]],
                 device const float* x       [[buffer(1)]],
                 device const int* positions [[buffer(2)]],
                 constant float& theta       [[buffer(3)]],
                 constant uint& n_heads      [[buffer(4)]],
                 constant uint& d            [[buffer(5)]],
                 constant uint& n_total      [[buffer(6)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid >= n_total) return;
    uint half_d = d / 2;
    uint i = gid % half_d;                     // pair index
    uint th = gid / half_d;                    // flat (token, head)
    uint t = th / n_heads;
    float freq = 1.0f / precise::pow(theta, float(i) / float(half_d));
    float ang = float(positions[t]) * freq;
    float c = precise::cos(ang), s = precise::sin(ang);
    uint base = th * d;
    float a = x[base + i], b = x[base + i + half_d];
    out[base + i] = a * c - b * s;
    out[base + i + half_d] = a * s + b * c;
}
