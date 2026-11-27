#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ingine::kernels::nvfp4 {

__device__ __forceinline__ uint32_t f16x2_from_e4m3x2(uint16_t v) {
    uint32_t r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(v));
    return r;
}

__device__ __forceinline__ void f16x2_from_e2m1x4(uint32_t w, uint32_t (&h)[4]) {
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(w));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Decodes a chunk into 16 bf16x2 words (element pairs 0..15) holding exactly decode_experts' values.
__device__ __forceinline__ void decode(uint4 codes, uint16_t scales, float factor, uint32_t (&o)[16]) {
    const uint32_t s2 = f16x2_from_e4m3x2(scales);
    const __half2 sc = *reinterpret_cast<const __half2*>(&s2);
    const __half2 slo = __low2half2(sc), shi = __high2half2(sc);
    const uint32_t words[4] = {codes.x, codes.y, codes.z, codes.w};
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        uint32_t h[4];
        f16x2_from_e2m1x4(words[w], h);
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            // code * scale is exact in f16 (<= 5 significant bits, |v| in [2^-10, 2688]); the factor
            // multiply rounds once in fp32, then to bf16, as in decode_experts.
            const __half2 cs = __hmul2(*reinterpret_cast<const __half2*>(&h[b]), w < 2 ? slo : shi);
            const float2 v = __half22float2(cs);
            o[4 * w + b] = pack_bf16(v.x * factor, v.y * factor);
        }
    }
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

}  // namespace ingine::kernels::nvfp4
