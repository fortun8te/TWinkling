// Non-cluster split-K attention for the sliding (1024-slot ring, BF16) layers on RTX 5080, verify widths 1..8.
// The cluster kernel (8 CTAs per KV head: 64 CTAs on 84 SMs) spends most of a call in its prologue and tile
// barriers over only 128 keys per CTA; here the work is three stream-ordered, graph-capturable kernels:
//   1. ring_prep   (8 CTAs): q / rel post-processing, new K/V conv + clip / norm into the workspace, conv states.
//   2. ring_main   ((NS + 1) x 8 CTAs): split s < NS takes ring slots [s * SL, +SL), split NS the call's own
//      keys; per query row the split's (max, sum, sum e * v) in the log2 domain.
//   3. ring_reduce (8 x M CTAs): merges the splits with the sink, applies the output gate; then commits the new
//      keys to ring slots (p0 + m) % 1024 and kpos (after every split has read the old ones).
// Semantics follow the FP64 oracle in tests/kernels/test_attn.cpp (sliding branch).
#pragma once

namespace ring_splitk {

constexpr int HQ = 24, HKV = 8, D = 128, DREL = 16, EXT = 1024, WIN = 1024, GQA = 3, C = HKV * D;
constexpr int OFF_GATE = 3072, OFF_REL = 6144, OFF_SINK = 6528, OFF_K = 6552, OFF_V = 7576;
constexpr int R = 8600, MAXM = 8, NT = 256;
constexpr int NS = 20, SL = WIN / NS + 1;  // 52 slots per split (last split shorter)
constexpr int ROWS_MAX = GQA * MAXM;      // query rows per KV head
constexpr float CLIP_QK = 22.627416997969522f, CLIP_REL = 8.f;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float QS2 = 0.08838834764831845f * LOG2E;  // 1/sqrt(128) in the log2 domain

// workspace layout (bytes)
constexpr size_t WS_Q = 256;  // past the cluster kernel's counter                                            // fp32 [HKV][24][D] (scaled by QS2)
constexpr size_t WS_REL = WS_Q + size_t(HKV) * ROWS_MAX * D * 4;         // fp32 [HKV][24][16]
constexpr size_t WS_NK = WS_REL + size_t(HKV) * ROWS_MAX * DREL * 4;     // bf16 [MAXM][HKV][D]
constexpr size_t WS_NV = WS_NK + size_t(MAXM) * C * 2;                   // bf16 [MAXM][HKV][D]
constexpr size_t WS_PART = WS_NV + size_t(MAXM) * C * 2;                 // fp32 [HKV][NS+1][24][D + 2]
constexpr size_t PSTR = D + 2;
constexpr size_t workspace_bytes() { return WS_PART + size_t(HKV) * (NS + 1) * ROWS_MAX * PSTR * 4; }

struct Params {
    const float* proj;  // [M, R]
    int M;
    const float* q_scale;
    const float* r_scale;
    const float* rel_proj;  // [16, 1024]
    const float* k_conv;    // [C, 4]
    const float* v_conv;
    float* conv_k;          // [3, C]
    float* conv_v;
    __nv_bfloat16* k_cache;  // [1024, 8, 128]
    __nv_bfloat16* v_cache;
    long long* kpos;         // [1024]
    const long long* pos;
    uint8_t* ws;
    __nv_bfloat16* out;      // [M, 3072]
};

__device__ __forceinline__ float rbf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}
__device__ __forceinline__ float wmax(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}
__device__ __forceinline__ float clip_factor(float ss, float t) {
    const float norm = ss > 0.f ? sqrtf(ss) : 0.f;
    return fminf(1.f, t / (norm + 1e-6f));
}

// ---------------------------------------------------------------------------------- 1: prep (block = KV head g)
__global__ void __launch_bounds__(NT) ring_prep_kernel(const Params p) {
    const int g = blockIdx.x, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const int M = p.M;
    float* wq = reinterpret_cast<float*>(p.ws + WS_Q) + size_t(g) * ROWS_MAX * D;
    float* wr = reinterpret_cast<float*>(p.ws + WS_REL) + size_t(g) * ROWS_MAX * DREL;
    // q rows (warp per row): rbf(clip(proj * q_scale)) * QS2
    for (int row = warp; row < GQA * M; row += NT / 32) {
        const int m = row / GQA, i = row % GQA, h = g * GQA + i;
        float x[4], ss = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int d = lane * 4 + e;
            x[e] = p.proj[size_t(m) * R + h * D + d] * p.q_scale[h * D + d];
            ss += x[e] * x[e];
        }
        const float f = clip_factor(wsum(ss), CLIP_QK);
#pragma unroll
        for (int e = 0; e < 4; ++e) wq[size_t(row) * D + lane * 4 + e] = rbf(x[e] * f) * QS2;
        // rel row: 16 values on lanes 0..15
        float r = lane < DREL ? p.proj[size_t(m) * R + OFF_REL + h * DREL + lane] * p.r_scale[h * DREL + lane] : 0.f;
        const float fr = clip_factor(wsum(r * r), CLIP_REL);
        if (lane < DREL) wr[row * DREL + lane] = rbf(r * fr);
    }
    // new K/V: static conv (ext = 3 state values then the call's inputs), / sqrt(5), BF16
    __shared__ float nkv[2][MAXM][D];
    for (int idx = t; idx < 2 * D; idx += NT) {
        const int kv = idx / D, ch = idx % D, c = g * D + ch;
        const float* st = kv ? p.conv_v : p.conv_k;
        const float* w = kv ? p.v_conv : p.k_conv;
        float ext[3 + MAXM];
#pragma unroll
        for (int j = 0; j < 3; ++j) ext[j] = st[j * C + c];
#pragma unroll
        for (int m = 0; m < MAXM; ++m) ext[3 + m] = m < M ? p.proj[size_t(m) * R + (kv ? OFF_V : OFF_K) + c] : 0.f;
        constexpr float inv_sqrt5 = 1.f / 2.2360679774997896f;
#pragma unroll
        for (int m = 0; m < MAXM; ++m) {
            if (m < M) {
                float a = __fmul_rn(w[c * 4 + 0], ext[m]);
                a = __fadd_rn(a, __fmul_rn(w[c * 4 + 1], ext[m + 1]));
                a = __fadd_rn(a, __fmul_rn(w[c * 4 + 2], ext[m + 2]));
                a = __fadd_rn(a, __fmul_rn(w[c * 4 + 3], ext[m + 3]));
                nkv[kv][m][ch] = rbf(__fmul_rn(__fadd_rn(a, ext[m + 3]), inv_sqrt5));
            }
        }
        float* sd = kv ? p.conv_v : p.conv_k;
#pragma unroll
        for (int j = 0; j < 3; ++j)
#pragma unroll
            for (int m = 1; m <= MAXM; ++m)
                if (m == M) sd[j * C + c] = ext[m + j];
    }
    __syncthreads();
    // clip keys, RMS-normalise values (warp per (kv, token)), BF16 into the workspace
    __nv_bfloat16* nk = reinterpret_cast<__nv_bfloat16*>(p.ws + WS_NK);
    __nv_bfloat16* nv = reinterpret_cast<__nv_bfloat16*>(p.ws + WS_NV);
    for (int pr = warp; pr < 2 * M; pr += NT / 32) {
        const int kv = pr / M, m = pr % M;
        float x[4], ss = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) x[e] = nkv[kv][m][lane * 4 + e], ss += x[e] * x[e];
        ss = wsum(ss);
        const float f = kv == 0 ? clip_factor(ss, CLIP_QK) : rsqrtf(ss / float(D) + 1e-6f);
        __nv_bfloat16* dst = (kv ? nv : nk) + (size_t(m) * HKV + g) * D;
#pragma unroll
        for (int e = 0; e < 4; ++e) dst[lane * 4 + e] = __float2bfloat16_rn(x[e] * f);
    }
}

// ---------------------------------------------------------------------------------- 2: main
// block (split s, head g); key list: ring slots of the split (old entries valid per token) or the new keys
__global__ void __launch_bounds__(NT) ring_main_kernel(const Params p) {
    const int s = blockIdx.x, g = blockIdx.y, t = threadIdx.x;
    const int M = p.M, ROWS = GQA * M;
    const long long p0 = *p.pos;
    __shared__ float qs[ROWS_MAX][D + 1];  // +1 / +2 padding: thread-varying rows and keys hit distinct banks
    __shared__ float rels[ROWS_MAX][DREL];
    __shared__ __align__(16) __nv_bfloat16 ks[SL][D + 2], vs[SL][D];
    __shared__ int kp_s[SL];
    __shared__ float sc[ROWS_MAX][SL];  // scores (log2 domain), -inf when masked
    const float* wq = reinterpret_cast<const float*>(p.ws + WS_Q) + size_t(g) * ROWS_MAX * D;
    const float* wr = reinterpret_cast<const float*>(p.ws + WS_REL) + size_t(g) * ROWS_MAX * DREL;
    for (int i = t; i < ROWS * D; i += NT) qs[i / D][i % D] = wq[i];
    for (int i = t; i < ROWS * DREL; i += NT) rels[i / DREL][i % DREL] = wr[i];
    const bool own = s == NS;  // the call's own keys
    const int first = own ? 0 : s * SL;
    const int n = own ? M : max(0, min(SL, WIN - first));
    for (int i = t; i < n * (D / 8); i += NT) {
        const int k = i / (D / 8), c = (i % (D / 8)) * 8;
        const __nv_bfloat16* ksrc = own ? reinterpret_cast<const __nv_bfloat16*>(p.ws + WS_NK) + (size_t(k) * HKV + g) * D
                                        : p.k_cache + (size_t(first + k) * HKV + g) * D;
        const __nv_bfloat16* vsrc = own ? reinterpret_cast<const __nv_bfloat16*>(p.ws + WS_NV) + (size_t(k) * HKV + g) * D
                                        : p.v_cache + (size_t(first + k) * HKV + g) * D;
        {
            const uint4 kk = *reinterpret_cast<const uint4*>(ksrc + c);
            const uint32_t* kw = reinterpret_cast<const uint32_t*>(&kk);
#pragma unroll
            for (int q = 0; q < 4; ++q) *reinterpret_cast<uint32_t*>(&ks[k][c + 2 * q]) = kw[q];
        }
        *reinterpret_cast<uint4*>(&vs[k][c]) = *reinterpret_cast<const uint4*>(vsrc + c);
    }
    for (int k = t; k < n; k += NT) kp_s[k] = own ? int(p0 + k) : int(p.kpos[first + k]);
    __syncthreads();
    // scores: thread per (row, key)
    for (int i = t; i < ROWS * n; i += NT) {
        const int row = i / n, k = i % n, m = row / GQA;
        const long long pt = p0 + m;
        const long long kp = kp_s[k];
        bool ok;
        long long dist;
        if (own) {
            ok = k <= m;
            dist = m - k;
        } else {
            ok = kp >= 0 && kp < p0 && pt - kp < WIN;
            dist = pt - kp;
        }
        float v = -INFINITY;
        if (ok) {
            float dot = 0.f;
#pragma unroll 8
            for (int d = 0; d < D; d += 2) {
                const float2 kk = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&ks[k][d]));
                dot = fmaf(qs[row][d], kk.x, fmaf(qs[row][d + 1], kk.y, dot));
            }
            float rl = 0.f;
            if (dist < EXT) {
#pragma unroll
                for (int j = 0; j < DREL; ++j) rl = fmaf(rels[row][j], p.rel_proj[size_t(j) * EXT + dist], rl);
                rl = rbf(rl);
            }
            v = dot + rl * LOG2E;
        }
        sc[row][k] = v;
    }
    __syncthreads();
    // per row: max, sum of exp2, sum e * v (warp per row; lanes over 4 dims each)
    float* part = reinterpret_cast<float*>(p.ws + WS_PART) + (size_t(g) * (NS + 1) + s) * ROWS_MAX * PSTR;
    const int warp = t >> 5, lane = t & 31;
    for (int row = warp; row < ROWS; row += NT / 32) {
        float mx = -INFINITY;
        for (int k = lane; k < n; k += 32) mx = fmaxf(mx, sc[row][k]);
        mx = wmax(mx);
        float o[4] = {0.f, 0.f, 0.f, 0.f}, sum = 0.f;
        if (mx > -INFINITY) {
            for (int k = 0; k < n; ++k) {
                const float e = exp2f(sc[row][k] - mx);  // exp2(-inf) = 0
                sum += e;
#pragma unroll
                for (int q = 0; q < 4; ++q) o[q] = fmaf(e, __bfloat162float(vs[k][lane * 4 + q]), o[q]);
            }
        }
        float* pr = part + size_t(row) * PSTR;
#pragma unroll
        for (int q = 0; q < 4; ++q) pr[lane * 4 + q] = o[q];
        if (lane == 0) pr[D] = mx, pr[D + 1] = sum;
    }
}

// ---------------------------------------------------------------------------------- 3: reduce + commit
// block (head g, token m): 3 query rows; then (head g) the new K/V row of token m into the ring
__global__ void __launch_bounds__(NT) ring_reduce_kernel(const Params p) {
    const int g = blockIdx.x, m = blockIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const long long p0 = *p.pos;
    const float* part = reinterpret_cast<const float*>(p.ws + WS_PART);
    if (warp < GQA) {
        const int row = m * GQA + warp, h = g * GQA + warp;
        const float sink = p.proj[size_t(m) * R + OFF_SINK + h] * LOG2E;
        float mx = sink;
        for (int s = 0; s <= NS; ++s) mx = fmaxf(mx, part[((size_t(g) * (NS + 1) + s) * ROWS_MAX + row) * PSTR + D]);
        float den = exp2f(sink - mx), o[4] = {0.f, 0.f, 0.f, 0.f};
        for (int s = 0; s <= NS; ++s) {
            const float* pr = part + ((size_t(g) * (NS + 1) + s) * ROWS_MAX + row) * PSTR;
            const float pm = pr[D];
            if (!(pm > -INFINITY)) continue;
            const float f = exp2f(pm - mx);
            den += f * pr[D + 1];
#pragma unroll
            for (int q = 0; q < 4; ++q) o[q] = fmaf(f, pr[lane * 4 + q], o[q]);
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int d = lane * 4 + q;
            const float gate = 1.f / (1.f + expf(-p.proj[size_t(m) * R + OFF_GATE + h * D + d]));
            p.out[size_t(m) * HQ * D + h * D + d] = __float2bfloat16_rn(rbf(o[q] / den) * gate);
        }
    }
    // commit this token's new key/value row of head g; kpos once per token
    const int slot = int((p0 + m) & (WIN - 1));
    const __nv_bfloat16* nk = reinterpret_cast<const __nv_bfloat16*>(p.ws + WS_NK) + (size_t(m) * HKV + g) * D;
    const __nv_bfloat16* nv = reinterpret_cast<const __nv_bfloat16*>(p.ws + WS_NV) + (size_t(m) * HKV + g) * D;
    if (t < D) {
        p.k_cache[(size_t(slot) * HKV + g) * D + t] = nk[t];
        p.v_cache[(size_t(slot) * HKV + g) * D + t] = nv[t];
    }
    if (g == 0 && t == 0) p.kpos[slot] = p0 + m;
}

inline void attn_decode_ring(const ingine::kernels::attn::DecodeArgs& a, cudaStream_t stream) {
    Params p;
    p.proj = a.proj;
    p.M = a.M;
    p.q_scale = a.q_scale;
    p.r_scale = a.r_scale;
    p.rel_proj = a.rel_proj;
    p.k_conv = a.k_conv;
    p.v_conv = a.v_conv;
    p.conv_k = a.conv_k_state;
    p.conv_v = a.conv_v_state;
    p.k_cache = static_cast<__nv_bfloat16*>(a.k_cache);
    p.v_cache = static_cast<__nv_bfloat16*>(a.v_cache);
    p.kpos = reinterpret_cast<long long*>(a.kpos);
    p.pos = reinterpret_cast<const long long*>(a.pos);
    p.ws = static_cast<uint8_t*>(a.workspace);
    p.out = a.out;
    ring_prep_kernel<<<HKV, NT, 0, stream>>>(p);
    ring_main_kernel<<<dim3(NS + 1, HKV), NT, 0, stream>>>(p);
    ring_reduce_kernel<<<dim3(HKV, a.M), NT, 0, stream>>>(p);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace ring_splitk
