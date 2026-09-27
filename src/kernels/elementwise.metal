#include <metal_stdlib>
using namespace metal;

// SwiGLU core: out[i] = silu(gate[i]) * up[i]. One thread per element.
kernel void silu_mul(device float* out        [[buffer(0)]],
                     device const float* gate [[buffer(1)]],
                     device const float* up   [[buffer(2)]],
                     constant uint& n         [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float g = gate[i];
    out[i] = g / (1.0f + precise::exp(-g)) * up[i];
}
