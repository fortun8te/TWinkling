// Attention core of one Inkling-2-Mini layer visit for M = 1..8 consecutive decode tokens, in ONE kernel.
// Contract: native/KERNELS.md, area `attn`.
//
// Design
// ------
// * Work: one thread-block cluster of CS = 16 blocks per KV head (grid 16 x 8 = 128 blocks). The cached
//   keys [0, L_old) (full layer: positions 0..p0-1 of the linear cache; sliding layer: the 1024 ring slots
//   with their kpos) are dealt to the 16 blocks in tiles, round robin, so every block gets an equal share
//   of the expensive relative-bias tail (distance < 1024). Block 0 (the "owner") also takes the call's M
//   new keys. Everything is sized from `pos` on the device, so the launch shape is fixed for a CUDA graph.
// * At short contexts the kernel is bound by dependent round trips, not bytes, so it has few of them:
//   (1) every input that does not depend on `pos` (q / rel / gate / sink rows of `proj` and their scales,
//   the owner's conv inputs and rel_proj[:, 0..7]) is fetched with cp.async into shared memory: one round
//   trip. M = 1 issues the K/V stream (first two buffers, all of a sliding layer's keys) before the PDL
//   wait, so it is in flight during proj's tail and the staged round trip is not queued behind it (K/V
//   issued after the wait but ahead of the staged inputs delays them by ~2 us; after them it costs a
//   round trip, 0.5 us per sliding visit in the decode step); M > 1 issues its first tile after the staged
//   inputs land. (2) The rel_proj column of each key is fetched together with its K/V for the distance a
//   full, in-order ring (or the linear cache) implies, and re-read only if the actual kpos disagrees.
//   (3) q / rel post-processing (scale, L2 clip, BF16, log scaling) is done once per block by a few
//   warps, branch-free, and shared through shared memory. (4) Split partials (max, sum, unnormalised o)
//   go to the workspace (L2); one cluster barrier;
//   block r then produces output dims 8r..8r+7 of every row (sink logit, sigmoid gate, BF16) from the 16
//   partials. (Distributed shared memory was measured slower here: ~1 GB/s per SM on this GPU.)
//   (5) Programmatic dependent launch: weights are staged before griddepcontrol.wait, and dependents are
//   released as soon as the block's first K/V loads are issued (all 128 blocks are resident by then, so
//   the next kernel, proj_conv, only takes free SM resources and streams its weights while this kernel
//   is latency-bound, its requests queued behind ours; it still waits for this grid's completion before
//   reading our output). What the kernel right before (proj) writes, and what this kernel writes for its
//   own tokens (conv states, cache rows >= p0, workspace), is read after the wait only: cp.async /
//   coherent `asm volatile` loads with memory clobbers, which the compiler cannot hoist. `pos` (any M),
//   kpos and the cache rows below p0 (M = 1) are read before it: their writers ran two or more kernels
//   back, and those are complete once this grid runs, because it starts only after every proj block has
//   passed its own griddepcontrol.wait (gemv triggers after it), which returns once the kernel before
//   proj, and hence every earlier kernel, has completed. This holds as long as attention follows a kernel
//   that triggers only after its own wait. The wait is fenced (a loop over p.zero): ptxas sinks loads
//   below a plain one.
// * Main loop, M = 1: warp = key slice (8 warps). A half-warp owns one key; lane c of the half holds dims
//   8c..8c+7, so each K and V row (256 B) is read with 16-byte vector loads (coherent, no L1
//   allocation). Two register buffers of 2 key pairs per warp alternate (one in flight while the other
//   is consumed): 16-32 KB in flight per block, enough for DRAM bandwidth with 128 blocks. Scores are
//   reduced over the 16 lanes with xor shuffles; the bias rel . rel_proj[:, d] is reduced the same way
//   and rounded to BF16, and skipped warp-uniformly when no key of the warp is within 1024. Online
//   softmax in the log2 domain (q is pre-scaled by log2(e)/sqrt(128)), FP32 accumulators.
// * Main loop, M > 1, on tensor cores: the block loads each tile of keys once with cp.async into shared
//   memory (own region, the first tiles issued before q / rel processing). Rows r = 3 token + head
//   (8-padded). S^T = K Q^T with mma.m16n8k16 bf16 (exact bf16 q, k; fp32 sums; 1/sqrt(128) applied to the
//   sums), (16 keys, 8 rows) items dealt over the warps; mask and relative bias per element; per-item row
//   max and P = exp2(S - max) stored as fp16 hi + lo (P - hi). Then O^T += V^T P^T with mma.m16n8k16 fp16
//   (V in fp16, exact for RMS-normed values), warp w owning dims 16w..16w+15, each item rescaled to the
//   running row max in registers. The owner runs the call's own keys first as a one-m-tile tile.
//   BF16 caches: 32-key tiles, double-buffered, with kpos and the rel_proj window; V converted in place.
//   E4M3 caches: 64-key tiles, two raw E4M3 tiles in flight; K mma fragments are read straight from the
//   raw rows (4 bytes per row and k-step, widened in registers, the k order inside an mma permuted and q
//   read with the same permutation), only V is widened to fp16 (a block pass into the staging region),
//   and the next tile is issued as soon as the S items are done. Measured on the full-layer verify
//   shapes (131K keys): M = 2 227 -> 169 us, M = 4 274 -> 191 us, M = 8 319 -> 286 us (32-key tiles,
//   one in flight, K and V widened, before).
// * The owner runs the key/value short convolutions of its head (+ key clip / value RMS norm) into shared
//   memory; it is the only reader of those conv-state channels. Its global writes (conv states, new
//   cache rows) come at the end. Sliding layers, M = 1: the ring slot and kpos it overwrites hold the
//   entry at distance 1024 (invalid for this token either way; a kpos >= p0 is ignored too), so the
//   other heads may read either value. M > 1: the overwritten slots still hold keys earlier tokens of
//   the call must see: the owner writes them after the cluster barrier (its head is done reading) and
//   the last head to get there (global counter in the workspace, reset by itself) writes kpos. A head
//   takes its turn right after its barrier, on a thread without output work, so the counter's round
//   trip overlaps the output phase instead of ending the kernel (-0.5 us per call).
// * Ring semantics: a cached ring entry is used iff 0 <= kpos < p0 and p - kpos < 1024; entries at
//   kpos >= p0 (left over from rejected speculative tokens) are ignored, as if overwritten first.
//   Values of masked keys are never multiplied in, so unused cache slots may hold anything.
// * FP8 full-layer caches (template F8; sliding rings are BF16 only): E4M3 without scale. The owner rounds
//   its new keys / values to BF16 and then to E4M3 (round to nearest even, saturated to +-448) in shared
//   memory, so the call's own keys are attended with exactly the values the cache will hold, and writes the
//   E4M3 bytes at the end. M = 1 reads 8-byte vectors with the same lanes, dims and buffer depth (half the
//   bytes in flight: the M = 1 kernels already use 128 registers). M > 1: see the main loop above.
// * Capturable: no allocation, no host sync, current stream; positions come from `pos` on the device.
// * Positions are int32 on the device (row offsets 64-bit); the host caps full caches at 2^30 rows.
// Ported from engine/csrc/attn/attn.cu (kernel bodies unchanged; torch launcher -> raw pointers).
#include "kernels/attn/attn.h"

#include "kernels/common/l2.cuh"

#include "core/device.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ingine::kernels::attn {
// RTX 5080 port: the kernels are compiled for 16-block clusters (RTX 5090) and 8-block clusters (GPUs whose GPCs
// cannot hold 16 co-resident blocks, e.g. the RTX 5080's 12-SM GPCs); the launcher picks 16 when it is schedulable.
namespace cs16 {
constexpr int CS = 16;
#include "kernels/attn/attn_impl.cuh"
}  // namespace cs16
namespace cs8 {
constexpr int CS = 8;
#include "kernels/attn/attn_impl.cuh"
}  // namespace cs8

namespace {
bool use16() {
    static const bool v = [] {
        for (int w : {0, 1024})
            for (int m = 1; m <= 8; ++m)
                if (cs16::max_active_clusters_impl(m, w) <= 0) return false;
        return true;
    }();
    return v;
}
}  // namespace

int cluster_size() { return use16() ? 16 : 8; }

std::size_t workspace_bytes(int max_m) {
    return std::max(cs16::workspace_bytes_impl(max_m), cs8::workspace_bytes_impl(max_m));
}

int max_active_clusters(int M, int window) {
    return use16() ? cs16::max_active_clusters_impl(M, window) : cs8::max_active_clusters_impl(M, window);
}

void attn_decode(const DecodeArgs& a, cudaStream_t stream) {
    if (use16()) cs16::attn_decode_impl(a, stream);
    else cs8::attn_decode_impl(a, stream);
}

}  // namespace ingine::kernels::attn
