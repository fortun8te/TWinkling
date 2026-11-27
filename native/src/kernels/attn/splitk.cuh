// Non-cluster Split-K attention for full-context FP8 / V4 layers on RTX 5080 (84 SMs).
// Eliminates the 8-block hardware cluster limit (64 blocks on 84 SMs, 85 KiB smem -> 1 block/SM)
// by splitting into three stream-ordered, graph-capturable kernels:
//   1. splitk_prep_kernel  (8 blocks): Q/REL post-processing + new-token conv/norm/quantize/KV-store.
//   2. splitk_main_kernel  (NS x 8 blocks, <40 KiB smem -> 2 blocks/SM on all 84 SMs): tiled K/V attention.
//   3. splitk_reduce_kernel (8 x M blocks): merges NS split partials from L2 with sink + output gate.
#pragma once

namespace splitk {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, GQA = 3, CONV_C = HKV * D;
constexpr int OFF_GATE = 3072, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576, OFF_KDYN = 8600,
              OFF_VDYN = 8632;
constexpr int R_FULL = 8664;
constexpr int MAXM = 8;
constexpr int NT = 256;
constexpr int NS = 21;  // 21 * 8 = 168 blocks = exact 2 blocks/SM wave on 84-SM RTX 5080
constexpr int PSTR = D + 4;
constexpr float CLIP_QK = 22.627416997969522f, CLIP_REL = 8.f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QSCALE = 0.08838834764831845f * LOG2E;
constexpr float NEG_INIT = -1e30f;
constexpr float INV_3 = 1.f / 3.f;

// Workspace layout (256-byte aligned regions):
//   [0, WS_QB): counter (unused here, kept for sliding compatibility)
//   [WS_QB, WS_RELD): QB [HKV][24][D] bf16 (48 KiB) -- aliased by Q_M1 [HKV][3][D] fp32 (12 KiB)
//   [WS_RELD, WS_PART): RELD [HKV][24][16] fp32 (12 KiB)
//   [WS_PART, ...): PART [HKV][NS][3*M][PSTR] fp32
constexpr size_t WS_QB = 256;
constexpr size_t WS_QB_BYTES = size_t(HKV) * 24 * D * 2;
constexpr size_t WS_RELD = WS_QB + WS_QB_BYTES;
constexpr size_t WS_RELD_BYTES = size_t(HKV) * 24 * DREL * 4;
constexpr size_t WS_PART = WS_RELD + WS_RELD_BYTES;
constexpr size_t workspace_bytes(int max_m) {
    return WS_PART + size_t(HKV) * NS * GQA * max_m * PSTR * 4;
}

struct Params {
    const float* proj;
    long long cap;
    int log_scaling;
    const float* q_scale;
    const float* r_scale;
    const float* rel_proj;
    const float* k_conv;
    const float* v_conv;
    float* conv_k;
    float* conv_v;
    void* k_cache;
    void* v_cache;
    const long long* pos;
    __nv_bfloat16* qb;
    float* reld;
    float* part;
    __nv_bfloat16* out;
};

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return unsigned(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float hsum16(float x) {
#pragma unroll
    for (int o = 1; o < 16; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}
__device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}
__device__ __forceinline__ float clip_factor(float ss, float t) {
    const float norm = ss > 0.f ? ss * rsqrtf(ss) : 0.f;
    return fminf(1.f, __fdividef(t, norm + 1e-6f));
}
__device__ __forceinline__ uint32_t pack2(float a, float b) {
    const __nv_bfloat162 v = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ float2 e4m3x2_f2(uint32_t pair) {
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t(pair), __NV_E4M3)));
}
__device__ __forceinline__ uint32_t e4m3x2_bf16x2(uint16_t pair) {
#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
    uint32_t result;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(result) : "h"(pair));
    return result;
#else
    const float2 f = e4m3x2_f2(pair);
    const __nv_bfloat162 value = __floats2bfloat162_rn(f.x, f.y);
    return *reinterpret_cast<const uint32_t*>(&value);
#endif
}
__device__ __forceinline__ uint32_t pack_e4m3x2(float a, float b) {
    return __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ void unpack8_fp8(const uint2 u, float f[8]) {
    float2 a = e4m3x2_f2(u.x & 0xffffu), b = e4m3x2_f2(u.x >> 16);
    f[0] = a.x, f[1] = a.y, f[2] = b.x, f[3] = b.y;
    a = e4m3x2_f2(u.y & 0xffffu), b = e4m3x2_f2(u.y >> 16);
    f[4] = a.x, f[5] = a.y, f[6] = b.x, f[7] = b.y;
}
__device__ __forceinline__ void unpack8_v4(const value4::Slice q, float f[8]) {
#pragma unroll
    for (int i = 0; i < 8; ++i) f[i] = value4::decode(q.nibble >> (4 * i), q.scale);
}

constexpr l2::Evict KV_L2 = l2::Evict::first;

__device__ __forceinline__ uint2 ld_kv8(const void* ptr, std::uint64_t pol) {
    uint2 v;
    if constexpr (l2::hinted<KV_L2>)
        asm volatile("ld.global.L1::no_allocate.L2::cache_hint.v2.u32 {%0,%1}, [%2], %3;"
                     : "=r"(v.x), "=r"(v.y) : "l"(ptr), "l"(pol) : "memory");
    else
        asm volatile("ld.global.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr) : "memory");
    return v;
}
__device__ __forceinline__ void cp16z(float* dst, const float* src, bool ok, std::uint64_t pol) {
    if constexpr (l2::hinted<KV_L2>)
        asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2, %3;" ::"r"(
                         unsigned(__cvta_generic_to_shared(dst))),
                     "l"(src), "r"(ok ? 16 : 0), "l"(pol) : "memory");
    else
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(unsigned(__cvta_generic_to_shared(dst))),
                     "l"(src), "r"(ok ? 16 : 0) : "memory");
}
__device__ __forceinline__ void cp4z(float* dst, const float* src, bool ok) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(unsigned(__cvta_generic_to_shared(dst))), "l"(src),
                 "r"(ok ? 4 : 0) : "memory");
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldsm_x2(uint32_t addr, uint32_t (&r)[2]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_f16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ------------------------------------------------------------------------- Kernel 1: Prep + KV store
template <int M, bool K4, bool V4>
__global__ void splitk_prep_kernel(const Params p) {
    constexpr int ROWS = GQA * M;
    constexpr int NRP = M > 1 ? (ROWS + 7) / 8 * 8 : 0;
    const int g = blockIdx.x;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, half = lane >> 4, c = lane & 15;
    const int p0 = int(*p.pos);
    if (p0 + M > p.cap) return;

    __shared__ float nkf[2 * M * D];

    auto tau_of = [&](int t) {
        return p.log_scaling ? 1.f + 0.1f * logf(fmaxf(float(p0 + t + 1) * (1.f / 8192.f), 1.f)) : 1.f;
    };

    // ---- Q and REL post-processing -> workspace (hot in L2)
    if constexpr (M == 1) {
        float* q_out = reinterpret_cast<float*>(p.qb) + size_t(g) * GQA * D;
        float* r_out = p.reld + size_t(g) * GQA * DREL;
        if (warp < 2 * GQA) {
            const int i = warp % GQA;
            const float tau = tau_of(0);
            if (warp < GQA) {
                const float4 a = reinterpret_cast<const float4*>(p.proj + g * GQA * D + i * D)[lane];
                const float4 s = reinterpret_cast<const float4*>(p.q_scale + g * GQA * D + i * D)[lane];
                float x[4] = {a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w};
                const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    x[e] = bf16r(x[e] * f);
                    if (p.log_scaling) x[e] = bf16r(x[e] * tau);
                }
                reinterpret_cast<float4*>(q_out + i * D)[lane] =
                    make_float4(x[0] * QSCALE, x[1] * QSCALE, x[2] * QSCALE, x[3] * QSCALE);
            } else {
                const float r = p.proj[OFF_REL + (g * GQA + i) * DREL + c] * p.r_scale[(g * GQA + i) * DREL + c];
                float y = bf16r(r * clip_factor(hsum16(r * r), CLIP_REL));
                if (p.log_scaling) y = bf16r(y * tau);
                if (half == 0) r_out[i * DREL + c] = y;
            }
        }
    } else {
        __nv_bfloat16* qb = p.qb + size_t(g) * NRP * D;
        float* reld = p.reld + size_t(g) * NRP * DREL;
#pragma unroll
        for (int k = 0; k < (ROWS + 7) / 8; ++k) {
            const int rq = warp + 8 * k, row = min(rq, ROWS - 1), t = row / GQA, i = row % GQA;
            const float tau = tau_of(t);
            const float4 a = reinterpret_cast<const float4*>(p.proj + size_t(t) * R_FULL + (g * GQA + i) * D)[lane];
            const float4 s = reinterpret_cast<const float4*>(p.q_scale + (g * GQA + i) * D)[lane];
            float x[4] = {a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w};
            const float f = clip_factor(wsum(x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]), CLIP_QK);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                x[e] = bf16r(x[e] * f);
                if (p.log_scaling) x[e] = bf16r(x[e] * tau);
            }
            if (rq < ROWS) {
                reinterpret_cast<uint2*>(qb + row * D + (((lane >> 1) ^ (row & 7)) * 8) + (lane & 1) * 4)[0] =
                    make_uint2(pack2(x[0], x[1]), pack2(x[2], x[3]));
            }
        }
#pragma unroll
        for (int k = 0; k < (ROWS + 15) / 16; ++k) {
            const int rr = warp * 2 + half + 16 * k, row = min(rr, ROWS - 1), t = row / GQA, i = row % GQA;
            const float r = p.proj[size_t(t) * R_FULL + OFF_REL + (g * GQA + i) * DREL + c] *
                            p.r_scale[(g * GQA + i) * DREL + c];
            float y = bf16r(r * clip_factor(hsum16(r * r), CLIP_REL));
            if (p.log_scaling) y = bf16r(y * tau_of(t));
            if (rr < ROWS) reld[row * DREL + c] = y;
        }
        for (int i = tid; i < (NRP - ROWS) * D / 2; i += NT)
            reinterpret_cast<uint32_t*>(qb + ROWS * D)[i] = 0u;
        for (int i = tid; i < (NRP - ROWS) * DREL; i += NT)
            reld[ROWS * DREL + i] = 0.f;
    }

    // ---- New K/V dynamic convolution + state update
    for (int idx = tid; idx < 2 * D; idx += NT) {
        const int kv = idx / D, ch = idx % D;
        const float* st_ptr = kv ? p.conv_v : p.conv_k;
        const float* w_ptr = kv ? p.v_conv : p.k_conv;
        float e[3 + M];
#pragma unroll
        for (int j = 0; j < 3; ++j) e[j] = st_ptr[j * CONV_C + g * D + ch];
#pragma unroll
        for (int m = 0; m < M; ++m) e[3 + m] = p.proj[size_t(m) * R_FULL + (kv ? OFF_V : OFF_K) + g * D + ch];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            float o = 0.f;
#pragma unroll
            for (int lag = 0; lag < 4; ++lag) {
                const float w = __fadd_rn(w_ptr[lag * CONV_C + g * D + ch],
                                          p.proj[size_t(m) * R_FULL + (kv ? OFF_VDYN : OFF_KDYN) + g * 4 + lag]);
                o = __fadd_rn(o, __fmul_rn(w, e[m + 3 - lag]));
            }
            nkf[(kv * M + m) * D + ch] = bf16r(__fmul_rn(__fadd_rn(o, e[m + 3]), INV_3));
        }
        float* st_dst = kv ? p.conv_v : p.conv_k;
#pragma unroll
        for (int i = 0; i < 3; ++i) st_dst[i * CONV_C + g * D + ch] = e[M + i];
    }
    __syncthreads();

    // ---- Norm + quantize + write new K/V directly to cache
    for (int pr = warp; pr < 2 * M; pr += NT / 32) {
        const int kv = pr / M, m = pr % M;
        const float4 x = reinterpret_cast<const float4*>(nkf + pr * D)[lane];
        const float ss = wsum(x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w);
        const float f = kv == 0 ? clip_factor(ss, CLIP_QK) : rsqrtf(ss / float(D) + 1e-6f);
        float y[4] = {bf16r(x.x * f), bf16r(x.y * f), bf16r(x.z * f), bf16r(x.w * f)};
        if (kv == 0) {
            if constexpr (K4) {
                auto row = value4::head(static_cast<char*>(p.k_cache), size_t(p0 + m) * HKV + g);
                value4::store(row, lane, y);
            } else {
                __nv_fp8_storage_t* dst = static_cast<__nv_fp8_storage_t*>(p.k_cache) + (size_t(p0 + m) * HKV + g) * D;
                reinterpret_cast<uint32_t*>(dst)[lane] =
                    pack_e4m3x2(y[0], y[1]) | (pack_e4m3x2(y[2], y[3]) << 16);
            }
        } else if constexpr (V4) {
            auto row = value4::head(static_cast<char*>(p.v_cache), size_t(p0 + m) * HKV + g);
            value4::store(row, lane, y);
        } else {
            __nv_fp8_storage_t* dst = static_cast<__nv_fp8_storage_t*>(p.v_cache) + (size_t(p0 + m) * HKV + g) * D;
            reinterpret_cast<uint32_t*>(dst)[lane] =
                pack_e4m3x2(y[0], y[1]) | (pack_e4m3x2(y[2], y[3]) << 16);
        }
    }
}

// ------------------------------------------------------------------------- Kernel 2: Split-K Main Loop
// M == 1: register-streaming path, zero dynamic shared memory -> maximum occupancy on all 84 SMs.
template <bool K4, bool V4>
__global__ void __launch_bounds__(NT, 2) splitk_main_m1_kernel(const Params p) {
    const int rank = blockIdx.x, g = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, half = lane >> 4, c = lane & 15;
    const int p0 = int(*p.pos);
    if (p0 + 1 > p.cap) return;
    const int L_total = p0 + 1;

    constexpr int KW = 8, U = 2;
    constexpr int STEP = KW * 2 * U, TILE = 2 * STEP, STRIDE = NS * TILE;
    const std::uint64_t kvpol = l2::policy<KV_L2>();

    const float* q_src = reinterpret_cast<const float*>(p.qb) + size_t(g) * GQA * D;
    const float* r_src = p.reld + size_t(g) * GQA * DREL;
    float qv[GQA][8], relv[GQA];
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
        const float4* qa = reinterpret_cast<const float4*>(q_src + i * D + c * 8);
        const float4 a0 = qa[0], a1 = qa[1];
        qv[i][0] = a0.x, qv[i][1] = a0.y, qv[i][2] = a0.z, qv[i][3] = a0.w;
        qv[i][4] = a1.x, qv[i][5] = a1.y, qv[i][6] = a1.z, qv[i][7] = a1.w;
        relv[i] = r_src[i * DREL + c];
    }

    using KType = std::conditional_t<K4, value4::Slice, uint2>;
    using VType = std::conditional_t<V4, value4::Slice, uint2>;
    struct Buf {
        KType k[U];
        VType v[U];
        int kp[U];
        float rp[U];
    };
    auto load = [&](Buf& b, const int base) {
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int j = base + (warp * U + u) * 2 + half;
            b.k[u] = {};
            b.v[u] = {};
            b.kp[u] = -1;
            b.rp[u] = 0.f;
            if (j < L_total) {
                const size_t off = (size_t(j) * HKV + g) * D + c * 8;
                if constexpr (K4)
                    b.k[u] = value4::load(value4::head(static_cast<const char*>(p.k_cache), size_t(j) * HKV + g), c);
                else
                    b.k[u] = ld_kv8(static_cast<const __nv_fp8_storage_t*>(p.k_cache) + off, kvpol);
                if constexpr (V4)
                    b.v[u] = value4::load(value4::head(static_cast<const char*>(p.v_cache), size_t(j) * HKV + g), c);
                else
                    b.v[u] = ld_kv8(static_cast<const __nv_fp8_storage_t*>(p.v_cache) + off, kvpol);
                b.kp[u] = j;
                const int d = p0 - j;
                if (d < EXT) b.rp[u] = __ldg(p.rel_proj + c * EXT + d);
            }
        }
    };

    float mrun[GQA], lrun[GQA], acc[GQA][8];
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
        mrun[i] = NEG_INIT, lrun[i] = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
    }

    auto compute = [&](const Buf& b) {
        float s[U][GQA];
        bool valid[U], need[U];
        bool any_need = false;
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int kp = b.kp[u], d = p0 - kp;
            const bool ok = kp >= 0;
            valid[u] = ok;
            need[u] = ok && d < EXT;
            any_need |= need[u];
            float kf[8];
            if constexpr (K4)
                unpack8_v4(ok ? b.k[u] : value4::Slice{}, kf);
            else
                unpack8_fp8(b.k[u], kf);
#pragma unroll
            for (int i = 0; i < GQA; ++i) {
                float t = 0.f;
#pragma unroll
                for (int e = 0; e < 8; ++e) t = fmaf(qv[i][e], kf[e], t);
                s[u][i] = hsum16(t);
            }
        }
        if (__any_sync(0xffffffffu, any_need)) {
#pragma unroll
            for (int u = 0; u < U; ++u) {
#pragma unroll
                for (int i = 0; i < GQA; ++i) {
                    const float bias = hsum16(relv[i] * b.rp[u]);
                    if (need[u]) s[u][i] = fmaf(bf16r(bias), LOG2E, s[u][i]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
            float mx = mrun[i];
#pragma unroll
            for (int u = 0; u < U; ++u) {
                if (!valid[u]) s[u][i] = -INFINITY;
                mx = fmaxf(mx, s[u][i]);
            }
            const float sc = exp2f(mrun[i] - mx);
            mrun[i] = mx;
            lrun[i] *= sc;
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[i][e] *= sc;
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            float vf[8];
            if constexpr (V4)
                unpack8_v4(valid[u] ? b.v[u] : value4::Slice{}, vf);
            else
                unpack8_fp8(valid[u] ? b.v[u] : uint2{}, vf);
#pragma unroll
            for (int i = 0; i < GQA; ++i) {
                const float pu = exp2f(s[u][i] - mrun[i]);
                lrun[i] += pu;
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[i][e] = fmaf(pu, vf[e], acc[i][e]);
            }
        }
    };

    const int c0 = rank * TILE;
    Buf A, B;
    if (c0 < L_total) load(A, c0);
    if (c0 + STEP < L_total) load(B, c0 + STEP);
    for (int base = c0; base < L_total; base += STRIDE) {
        compute(A);
        if (base + STRIDE < L_total) load(A, base + STRIDE);
        if (base + STEP < L_total) {
            compute(B);
            if (base + STRIDE + STEP < L_total) load(B, base + STRIDE + STEP);
        }
    }

    // Merge half-warps, then 8 warps in shared memory
#pragma unroll
    for (int i = 0; i < GQA; ++i) {
        const float mo = __shfl_xor_sync(0xffffffffu, mrun[i], 16);
        const float lo = __shfl_xor_sync(0xffffffffu, lrun[i], 16);
        const float mx = fmaxf(mrun[i], mo);
        const float f1 = exp2f(mrun[i] - mx), f2 = exp2f(mo - mx);
        lrun[i] = lrun[i] * f1 + lo * f2;
#pragma unroll
        for (int e = 0; e < 8; ++e)
            acc[i][e] = acc[i][e] * f1 + __shfl_xor_sync(0xffffffffu, acc[i][e], 16) * f2;
        mrun[i] = mx;
    }
    __shared__ float red_o[8 * GQA * D];
    __shared__ float red_ml[8 * GQA * 2];
    if (half == 0) {
#pragma unroll
        for (int i = 0; i < GQA; ++i) {
            float4* dst = reinterpret_cast<float4*>(red_o + (warp * GQA + i) * D + c * 8);
            dst[0] = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
            dst[1] = make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
            if (c == 0) red_ml[(warp * GQA + i) * 2] = mrun[i], red_ml[(warp * GQA + i) * 2 + 1] = lrun[i];
        }
    }
    __syncthreads();
    float* part = p.part + size_t(g * NS + rank) * GQA * PSTR;
    for (int idx = tid; idx < GQA * (D / 4); idx += NT) {
        const int row = idx / (D / 4), d4 = idx % (D / 4);
        float mx = NEG_INIT;
#pragma unroll
        for (int k = 0; k < KW; ++k) mx = fmaxf(mx, red_ml[(k * GQA + row) * 2]);
        float l = 0.f;
        float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int k = 0; k < KW; ++k) {
            const int w = k * GQA + row;
            const float f = exp2f(red_ml[w * 2] - mx);
            l += red_ml[w * 2 + 1] * f;
            const float4 a = reinterpret_cast<const float4*>(red_o + w * D)[d4];
            o.x += a.x * f, o.y += a.y * f, o.z += a.z * f, o.w += a.w * f;
        }
        __stcg(reinterpret_cast<float4*>(part + row * PSTR) + d4, o);
        if (d4 == 0) __stcg(reinterpret_cast<float2*>(part + row * PSTR + D), make_float2(mx, l));
    }
}

// Compact dynamic shared memory layout for M > 1 Split-K (<44 KiB for V4 and K4/V4 -> 2 blocks/SM on RTX 5080!).
template <int M, bool K4, bool V4>
struct MainSmem {
    static constexpr int ROWS = GQA * M;
    static constexpr int NRP = (ROWS + 7) / 8 * 8;
    static constexpr int KT = 32;
    static constexpr int NRAW = 3;
    static constexpr int RAW_K_STRIDE = K4 ? ((value4::kPackedBytes + sizeof(float) + 15) & ~15) : D;
    static constexpr int RAW_V_STRIDE = V4 ? ((value4::kPackedBytes + sizeof(float) + 15) & ~15) : D;
    static constexpr int RAW_TILE_BYTES = KT * (RAW_K_STRIDE + RAW_V_STRIDE);
    static constexpr int PSTRIDE = KT + 8;  // 40 fp16 = 80 B

    // Offsets in floats (16-byte aligned)
    static constexpr int WV = 0;                                      // [KT][D] fp16 -> KT*D/2 floats (8 KiB)
    static constexpr int WK = WV + KT * D / 2;                        // [KT][D] bf16 when K4 (8 KiB), else 0
    static constexpr int RAW = WK + (K4 ? KT * D / 2 : 0);            // NRAW * RAW_TILE_BYTES / 4 floats
    static constexpr int QB = (RAW + NRAW * RAW_TILE_BYTES / 4 + 3) & ~3;  // [NRP][D] bf16
    static constexpr int RELD = QB + NRP * D / 2;                     // [NRP][16] fp32
    static constexpr int PS = RELD + NRP * DREL;                      // [NRP][PSTRIDE] scaled fp16 (single-pass)
    static constexpr int ITM = PS + (NRP * PSTRIDE + 1) / 2;          // [KT/16][NRP] float2
    static constexpr int END = ITM + 2 * (KT / 16) * NRP;
    static constexpr size_t BYTES = size_t(END) * 4;
};

template <int M, bool K4, bool V4>
__global__ void __launch_bounds__(NT, 2) splitk_main_tc_kernel(const Params p) {
    using L = MainSmem<M, K4, V4>;
    constexpr int ROWS = L::ROWS, NRP = L::NRP, NTN = NRP / 8, KT = L::KT, KM = KT / 16;
    constexpr int NRAW = L::NRAW, RAWB = L::RAW_TILE_BYTES, PSTRIDE = L::PSTRIDE;

    const int rank = blockIdx.x, g = blockIdx.y;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int gid = lane >> 2, t4 = lane & 3;
    const int p0 = int(*p.pos);
    if (p0 + M > p.cap) return;
    const int L_total = p0 + M;

    extern __shared__ __align__(16) float sm[];
    __half* const wv = reinterpret_cast<__half*>(sm + L::WV);
    __nv_bfloat16* const wk = reinterpret_cast<__nv_bfloat16*>(sm + L::WK);
    char* const raw = reinterpret_cast<char*>(sm + L::RAW);
    __nv_bfloat16* const qb = reinterpret_cast<__nv_bfloat16*>(sm + L::QB);
    float* const reld = sm + L::RELD;
    __half* const ps = reinterpret_cast<__half*>(sm + L::PS);
    float2* const itm = reinterpret_cast<float2*>(sm + L::ITM);
    const std::uint64_t kvpol = l2::policy<KV_L2>();

    // Stage QB and RELD from L2 workspace into shared memory
    const __nv_bfloat16* qb_src = p.qb + size_t(g) * NRP * D;
    const float* reld_src = p.reld + size_t(g) * NRP * DREL;
    for (int i = tid; i < NRP * D / 8; i += NT)
        reinterpret_cast<uint4*>(qb)[i] = reinterpret_cast<const uint4*>(qb_src)[i];
    for (int i = tid; i < NRP * DREL / 4; i += NT)
        reinterpret_cast<float4*>(reld)[i] = reinterpret_cast<const float4*>(reld_src)[i];

    auto issue_raw = [&](const int t) {
        const int base = rank * KT + t * NS * KT;
        if (base < L_total) {
            char* const slot = raw + (t % NRAW) * RAWB;
            // K:
            if constexpr (K4) {
                if (tid < KT * 5) {
                    const int key = tid / 5, ch = tid % 5, j = base + key;
                    const bool ok = j < L_total;
                    const auto src = value4::head(static_cast<const char*>(p.k_cache), size_t(ok ? j : 0) * HKV + g);
                    char* dst = slot + key * L::RAW_K_STRIDE + ch * 16;
                    if (ch < 4)
                        cp16z(reinterpret_cast<float*>(dst), reinterpret_cast<const float*>(src.packed + ch * 16), ok,
                              kvpol);
                    else
                        cp4z(reinterpret_cast<float*>(dst), reinterpret_cast<const float*>(src.scale), ok);
                }
            } else {
                const int key = tid >> 3, ch = tid & 7, j = base + key;
                const bool ok = j < L_total;
                char* dst = slot + key * D + (ch ^ (key & 7)) * 16;
                cp16z(reinterpret_cast<float*>(dst),
                      reinterpret_cast<const float*>(static_cast<const __nv_fp8_storage_t*>(p.k_cache) +
                                                     (size_t(ok ? j : 0) * HKV + g) * D + ch * 16),
                      ok, kvpol);
            }
            // V:
            if constexpr (V4) {
                if (tid < KT * 5) {
                    const int key = tid / 5, ch = tid % 5, j = base + key;
                    const bool ok = j < L_total;
                    const auto src = value4::head(static_cast<const char*>(p.v_cache), size_t(ok ? j : 0) * HKV + g);
                    char* dst = slot + KT * L::RAW_K_STRIDE + key * L::RAW_V_STRIDE + ch * 16;
                    if (ch < 4)
                        cp16z(reinterpret_cast<float*>(dst), reinterpret_cast<const float*>(src.packed + ch * 16), ok,
                              kvpol);
                    else
                        cp4z(reinterpret_cast<float*>(dst), reinterpret_cast<const float*>(src.scale), ok);
                }
            } else {
                const int key = tid >> 3, ch = tid & 7, j = base + key;
                const bool ok = j < L_total;
                char* dst = slot + KT * L::RAW_K_STRIDE + key * D + ch * 16;
                cp16z(reinterpret_cast<float*>(dst),
                      reinterpret_cast<const float*>(static_cast<const __nv_fp8_storage_t*>(p.v_cache) +
                                                     (size_t(ok ? j : 0) * HKV + g) * D + ch * 16),
                      ok, kvpol);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };

    // Prime NRAW tiles in flight
#pragma unroll
    for (int t = 0; t < NRAW; ++t) issue_raw(t);

    auto roff = [](int key, int ch8) { return key * D + ((ch8 ^ (key & 7)) * 8); };
    auto half2_of = [](float a, float b) {
        a = fabsf(a) <= 65504.f ? a : 0.f;
        b = fabsf(b) <= 65504.f ? b : 0.f;
        const __half2 h = __floats2half2_rn(a, b);
        return *reinterpret_cast<const uint32_t*>(&h);
    };
    auto expand_k = [&](const int slot) {
        if constexpr (K4) {
            const char* tile = raw + slot * RAWB;
            constexpr int slices = D / 8;  // 16
            constexpr int chunks = KT * slices / NT;  // 2
#pragma unroll
            for (int n = 0; n < chunks; ++n) {
                const int i = tid + n * NT, key = i / slices, c8 = i % slices;
                const char* row = tile + key * L::RAW_K_STRIDE;
                const float scale = *reinterpret_cast<const float*>(row + value4::kPackedBytes);
                const float value = value4::decode(c8, scale);
                const uint32_t entry = __bfloat16_as_ushort(__float2bfloat16_rn(value));
                const uint32_t bits = reinterpret_cast<const uint32_t*>(row)[c8];
                uint32_t pairs[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const uint32_t code = bits >> (8 * e);
                    const uint32_t low = __shfl_sync(0xffffffffu, entry, int(code & 15u), slices);
                    const uint32_t high = __shfl_sync(0xffffffffu, entry, int((code >> 4) & 15u), slices);
                    pairs[e] = low | (high << 16);
                }
                *reinterpret_cast<uint4*>(wk + roff(key, c8)) = make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
            }
        }
    };
    auto expand_v = [&](const int slot) {
        const char* tile = raw + slot * RAWB + KT * L::RAW_K_STRIDE;
        if constexpr (V4) {
            constexpr int slices = D / 8;  // 16
            constexpr int chunks = KT * slices / NT;  // 2
#pragma unroll
            for (int n = 0; n < chunks; ++n) {
                const int i = tid + n * NT, key = i / slices, c8 = i % slices;
                const char* row = tile + key * L::RAW_V_STRIDE;
                const float scale = *reinterpret_cast<const float*>(row + value4::kPackedBytes);
                const float value = value4::decode(c8, scale);
                const uint32_t entry = __half_as_ushort(__float2half_rn(fabsf(value) <= 65504.f ? value : 0.f));
                const uint32_t bits = reinterpret_cast<const uint32_t*>(row)[c8];
                uint32_t pairs[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const uint32_t code = bits >> (8 * e);
                    const uint32_t low = __shfl_sync(0xffffffffu, entry, int(code & 15u), slices);
                    const uint32_t high = __shfl_sync(0xffffffffu, entry, int((code >> 4) & 15u), slices);
                    pairs[e] = low | (high << 16);
                }
                *reinterpret_cast<uint4*>(wv + roff(key, c8)) = make_uint4(pairs[0], pairs[1], pairs[2], pairs[3]);
            }
        } else {
            // 32 * 8 = 256 16-byte chunks -> 1 per thread
            const int key = tid >> 3, ch = tid & 7;
            const uint4 r = *reinterpret_cast<const uint4*>(tile + key * D + ch * 16);
            const uint32_t w[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t o[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float2 f = e4m3x2_f2((w[2 * h + e / 2] >> (16 * (e % 2))) & 0xffffu);
                    o[e] = half2_of(f.x, f.y);
                }
                *reinterpret_cast<uint4*>(wv + roff(key, 2 * ch + h)) = make_uint4(o[0], o[1], o[2], o[3]);
            }
        }
    };

    float oacc[NTN][4], mrow[NTN][2], lrow[NTN][2];
#pragma unroll
    for (int n = 0; n < NTN; ++n) {
#pragma unroll
        for (int v = 0; v < 4; ++v) oacc[n][v] = 0.f;
        mrow[n][0] = mrow[n][1] = NEG_INIT, lrow[n][0] = lrow[n][1] = 0.f;
    }

    constexpr int NW = NT / 32;  // 8
    // KM = 2, so NW / KM = 4 warps per key m-tile! Since NTN <= 3 (M <= 8), NPW = 1 ALWAYS!
    static_assert(NTN <= NW / KM);
    const int km = warp % KM, nn = warp / KM;

    int t = 0;
    for (int base = rank * KT; base < L_total; base += NS * KT, ++t) {
        asm volatile("cp.async.wait_group %0;" ::"n"(NRAW - 1) : "memory");
        __syncthreads();
        const char* kraw = raw + (t % NRAW) * RAWB;
        if constexpr (K4) {
            expand_k(t % NRAW);
            __syncthreads();
        }
        expand_v(t % NRAW);

        // Compute S = K Q^T while wv finishes
        if (nn < NTN) {
            float a4[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int ks = 0; ks < D / 16; ++ks) {
                uint32_t af[4];
                const int key = km * 16 + gid;
                if constexpr (K4) {
                    const int ch = 2 * ks + (t4 >> 1);
                    const uint2 k0 = *reinterpret_cast<const uint2*>(wk + roff(key, ch) + (t4 & 1) * 4);
                    const uint2 k1 = *reinterpret_cast<const uint2*>(wk + roff(key + 8, ch) + (t4 & 1) * 4);
                    af[0] = k0.x, af[1] = k1.x, af[2] = k0.y, af[3] = k1.y;
                } else {
                    const char* kr = kraw + key * D + ((ks ^ (key & 7)) * 16) + 4 * t4;
                    const uint32_t u0 = *reinterpret_cast<const uint32_t*>(kr);
                    const uint32_t u1 = *reinterpret_cast<const uint32_t*>(kr + 8 * D);
                    af[0] = e4m3x2_bf16x2(uint16_t(u0)), af[1] = e4m3x2_bf16x2(uint16_t(u1));
                    af[2] = e4m3x2_bf16x2(uint16_t(u0 >> 16)), af[3] = e4m3x2_bf16x2(uint16_t(u1 >> 16));
                }
                uint32_t bfr[2];
                const int row = nn * 8 + gid, ch = 2 * ks + (t4 >> 1);
                const uint2 q2 = *reinterpret_cast<const uint2*>(qb + row * D + ((ch ^ (row & 7)) * 8) + (t4 & 1) * 4);
                bfr[0] = q2.x, bfr[1] = q2.y;
                mma_bf16(a4, af, bfr);
            }
            const int r0 = nn * 8 + 2 * t4;
            float sv[4];
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int key = km * 16 + gid + (v >> 1) * 8, row = r0 + (v & 1), tok = row / GQA, j = base + key;
                const int d = p0 + tok - j;
                const bool ok = row < ROWS && j < L_total && d >= 0;
                float s = a4[v] * QSCALE;
                if (ok && d < EXT) {
                    const float* rl = reld + row * DREL;
                    float b = 0.f;
#pragma unroll
                    for (int k = 0; k < DREL; ++k) b = fmaf(rl[k], __ldg(p.rel_proj + k * EXT + d), b);
                    s = fmaf(bf16r(b), LOG2E, s);
                }
                sv[v] = ok ? s : -INFINITY;
            }
            float mx0 = fmaxf(sv[0], sv[2]), mx1 = fmaxf(sv[1], sv[3]);
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
            }
            mx0 = fmaxf(mx0, NEG_INIT), mx1 = fmaxf(mx1, NEG_INIT);
            float s0 = 0.f, s1 = 0.f;
            constexpr float kPScale = 32768.f;
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int key = km * 16 + gid + (v >> 1) * 8, row = r0 + (v & 1);
                const float pf = exp2f(sv[v] - ((v & 1) ? mx1 : mx0));
                ps[row * PSTRIDE + key] = __float2half_rn(pf * kPScale);
                ((v & 1) ? s1 : s0) += pf;
            }
#pragma unroll
            for (int o = 4; o < 32; o <<= 1) {
                s0 += __shfl_xor_sync(0xffffffffu, s0, o);
                s1 += __shfl_xor_sync(0xffffffffu, s1, o);
            }
            if (gid == 0) itm[km * NRP + r0] = make_float2(mx0, s0), itm[km * NRP + r0 + 1] = make_float2(mx1, s1);
        }
        __syncthreads();
        issue_raw(t + NRAW);  // raw slot (t % NRAW) is now free!

        float f[KM][NTN][2];
#pragma unroll
        for (int n = 0; n < NTN; ++n) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r = n * 8 + 2 * t4 + h;
                float2 it[KM];
                float mn = mrow[n][h];
#pragma unroll
                for (int k = 0; k < KM; ++k) {
                    it[k] = itm[k * NRP + r];
                    mn = fmaxf(mn, it[k].x);
                }
                const float sc = exp2f(mrow[n][h] - mn);
                float l = lrow[n][h] * sc;
#pragma unroll
                for (int k = 0; k < KM; ++k) {
                    f[k][n][h] = exp2f(it[k].x - mn);
                    l += it[k].y * f[k][n][h];
                }
                oacc[n][h] *= sc, oacc[n][h + 2] *= sc;
                mrow[n][h] = mn, lrow[n][h] = l;
            }
        }
#pragma unroll
        for (int ks = 0; ks < KM; ++ks) {
            uint32_t af[4];
            {
                const int q = lane >> 3, key = ks * 16 + (q >> 1) * 8 + (lane & 7);
                ldsm_x4_t(smem_u32(wv + roff(key, warp * 2 + (q & 1))), af);
            }
            constexpr float kPUnscale = 1.f / 32768.f;
#pragma unroll
            for (int n = 0; n < NTN; ++n) {
                uint32_t bfr[2];
                float acc_t[4] = {0.f, 0.f, 0.f, 0.f};
                const __half* pr = ps + (n * 8 + (lane & 7)) * PSTRIDE + ks * 16 + ((lane >> 3) & 1) * 8;
                ldsm_x2(smem_u32(pr), bfr);
                mma_f16(acc_t, af, bfr);
#pragma unroll
                for (int v = 0; v < 4; ++v) oacc[n][v] = fmaf(f[ks][n][v & 1] * kPUnscale, acc_t[v], oacc[n][v]);
            }
        }
    }

    // Drain any unused cp.async groups before writing partials
    asm volatile("cp.async.wait_all;" ::: "memory");

    float* part = p.part + size_t(g * NS + rank) * ROWS * PSTR;
#pragma unroll
    for (int n = 0; n < NTN; ++n) {
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const int row = n * 8 + 2 * t4 + (v & 1), dim = warp * 16 + gid + (v >> 1) * 8;
            if (row < ROWS) __stcg(part + row * PSTR + dim, oacc[n][v]);
        }
        if (warp == 0 && gid == 0) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const int row = n * 8 + 2 * t4 + v;
                if (row < ROWS)
                    __stcg(reinterpret_cast<float2*>(part + row * PSTR + D), make_float2(mrow[n][v], lrow[n][v]));
            }
        }
    }
}

// ------------------------------------------------------------------------- Kernel 3: Reduce NS partials
template <int M>
__global__ void splitk_reduce_kernel(const Params p) {
    constexpr int ROWS = GQA * M;
    const int g = blockIdx.x, m = blockIdx.y;
    const int dim = threadIdx.x;  // 0..127
    if (int(*p.pos) + M > p.cap) return;

#pragma unroll
    for (int h = 0; h < GQA; ++h) {
        const int row = m * GQA + h;
        const float* src = p.part + size_t(g * NS) * ROWS * PSTR + row * PSTR;
        float2 ml[NS];
        float o[NS];
#pragma unroll
        for (int q = 0; q < NS; ++q) {
            ml[q] = __ldcg(reinterpret_cast<const float2*>(src + q * ROWS * PSTR + D));
            o[q] = __ldcg(src + q * ROWS * PSTR + dim);
        }
        const float sink = p.proj[size_t(m) * R_FULL + OFF_SINK + g * GQA + h] * LOG2E;
        float mx = sink;
#pragma unroll
        for (int q = 0; q < NS; ++q) mx = fmaxf(mx, ml[q].x);
        float l = exp2f(sink - mx), a = 0.f;
#pragma unroll
        for (int q = 0; q < NS; ++q) {
            const float f = exp2f(ml[q].x - mx);
            l += ml[q].y * f;
            a += o[q] * f;
        }
        const float gate = 1.f / (1.f + expf(-p.proj[size_t(m) * R_FULL + OFF_GATE + (g * GQA + h) * D + dim]));
        p.out[size_t(m) * HQ * D + (g * GQA + h) * D + dim] = __float2bfloat16_rn(bf16r(a / l) * gate);
    }
}

template <int M, bool K4, bool V4>
void launch_m(const Params& p, cudaStream_t stream) {
    splitk_prep_kernel<M, K4, V4><<<dim3(HKV), dim3(NT), 0, stream>>>(p);
    if constexpr (M == 1) {
        splitk_main_m1_kernel<K4, V4><<<dim3(NS, HKV), dim3(NT), 0, stream>>>(p);
    } else {
        static const bool init = [] {
            CUDA_CHECK(cudaFuncSetAttribute(splitk_main_tc_kernel<M, K4, V4>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            int(MainSmem<M, K4, V4>::BYTES)));
            return true;
        }();
        (void)init;
        splitk_main_tc_kernel<M, K4, V4><<<dim3(NS, HKV), dim3(NT), MainSmem<M, K4, V4>::BYTES, stream>>>(p);
    }
    splitk_reduce_kernel<M><<<dim3(HKV, M), dim3(D), 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

template <bool K4, bool V4>
void dispatch(const Params& p, int M, cudaStream_t stream) {
    switch (M) {
        case 1: launch_m<1, K4, V4>(p, stream); break;
        case 2: launch_m<2, K4, V4>(p, stream); break;
        case 3: launch_m<3, K4, V4>(p, stream); break;
        case 4: launch_m<4, K4, V4>(p, stream); break;
        case 5: launch_m<5, K4, V4>(p, stream); break;
        case 6: launch_m<6, K4, V4>(p, stream); break;
        case 7: launch_m<7, K4, V4>(p, stream); break;
        default: launch_m<8, K4, V4>(p, stream); break;
    }
}

void attn_decode_splitk(const DecodeArgs& a, cudaStream_t stream) {
    Params p{};
    p.proj = a.proj;
    p.cap = a.cache_rows;
    p.log_scaling = a.log_scaling;
    p.q_scale = a.q_scale;
    p.r_scale = a.r_scale;
    p.rel_proj = a.rel_proj;
    p.k_conv = a.k_conv;
    p.v_conv = a.v_conv;
    p.conv_k = a.conv_k_state;
    p.conv_v = a.conv_v_state;
    p.k_cache = a.k_cache;
    p.v_cache = a.v_cache;
    p.pos = reinterpret_cast<const long long*>(a.pos);
    auto* ws = static_cast<char*>(a.workspace);
    p.qb = reinterpret_cast<__nv_bfloat16*>(ws + WS_QB);
    p.reld = reinterpret_cast<float*>(ws + WS_RELD);
    p.part = reinterpret_cast<float*>(ws + WS_PART);
    p.out = a.out;
    if (a.cache_k4) dispatch<true, true>(p, a.M, stream);
    else if (a.cache_v4) dispatch<false, true>(p, a.M, stream);
    else dispatch<false, false>(p, a.M, stream);
}

}  // namespace splitk
