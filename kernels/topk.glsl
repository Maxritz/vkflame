#version 460

layout(local_size_x = 256) in;

layout(set = 0, binding = 0) readonly  buffer XBuf      { float x_data[]; };
layout(set = 0, binding = 1) writeonly buffer ValBuf    { float out_vals[]; };
layout(set = 0, binding = 2) writeonly buffer IdxBuf    { uint  out_idx[]; };

layout(push_constant) uniform PC {
    uint M;       // rows
    uint N;       // columns
    uint K;       // number of top-k values to find (K <= MAX_K)
    uint largest; // 1 = largest (top-k), 0 = smallest (bottom-k)
} pc;

// LDS: one current-best slot per thread, plus the selected thread index.
shared float s_val[256];
shared uint  s_idx[256];
shared uint  s_best_tid;  // which thread's entry was selected this round

// Maximum K supported.  Private register arrays must have compile-time size.
#define MAX_K 50

void main() {
    uint row = gl_WorkGroupID.x;
    uint tid = gl_LocalInvocationID.x;

    if (row >= pc.M) return;

    uint row_base = row * pc.N;

    // ── Phase 1: per-thread sweep ────────────────────────────────────────
    // Each thread processes elements at indices {tid, tid+256, tid+512, ...}
    // and maintains a private sorted list (descending) of its top-K candidates.
    //
    // Correctness: for any N, the global top-K must appear somewhere in the
    // K-best-per-thread lists (each thread's stripe holds N/256 elements,
    // so tracking K of them covers any cluster of top-K in one stripe).

    float my_val[MAX_K];
    uint  my_idx[MAX_K];
    uint  my_k = 0u;  // valid entries in my_val/my_idx (0..K)

    for (uint i = tid; i < pc.N; i += 256u) {
        float raw = x_data[row_base + i];
        // For smallest-k: negate so the rest of the logic always optimises "largest".
        float v = (pc.largest == 0u) ? -raw : raw;

        // Skip if the list is full and v cannot displace our current worst.
        if (my_k == pc.K && v <= my_val[my_k - 1u])
            continue;

        // Insertion sort (list is sorted descending):
        // find first position where my_val[pos] < v.
        uint pos = (my_k < pc.K) ? my_k : pc.K - 1u;
        for (uint j = 0u; j < my_k && j < pc.K; j++) {
            if (my_val[j] < v) { pos = j; break; }
        }

        // Shift elements right from pos to make room.
        uint shift_end = (my_k < pc.K) ? my_k : pc.K - 1u;
        for (uint j = shift_end; j > pos; j--) {
            my_val[j] = my_val[j - 1u];
            my_idx[j] = my_idx[j - 1u];
        }
        my_val[pos] = v;
        my_idx[pos] = i;
        if (my_k < pc.K) my_k++;
    }

    // ── Phase 2: cross-thread merge ──────────────────────────────────────
    // Each thread exposes its current best candidate in LDS.
    // Thread 0 selects the global winner, writes to output, then the winning
    // thread advances to its next private candidate and refreshes LDS.
    // Repeat K times.

    s_val[tid] = (my_k > 0u) ? my_val[0u] : -1e38;
    s_idx[tid] = (my_k > 0u) ? my_idx[0u] : 0u;
    uint cur_ptr = 0u;  // next unread position in my_val for this thread
    barrier();
    memoryBarrierShared();

    for (uint k = 0u; k < pc.K; k++) {
        // Thread 0: linear scan for the global maximum in s_val[0..255].
        if (tid == 0u) {
            uint  best_t = 0u;
            float best_v = s_val[0u];
            for (uint t = 1u; t < 256u; t++) {
                if (s_val[t] > best_v) {
                    best_v = s_val[t];
                    best_t = t;
                }
            }
            // Undo the negation applied for smallest-k.
            float out_v = (pc.largest == 0u) ? -best_v : best_v;
            out_vals[row * pc.K + k] = out_v;
            out_idx [row * pc.K + k] = s_idx[best_t];
            s_val[best_t] = -1e38;  // invalidate — winning thread will refresh
            s_best_tid = best_t;
        }
        barrier();
        memoryBarrierShared();

        // The winning thread advances its pointer and updates its LDS slot.
        if (tid == s_best_tid) {
            cur_ptr++;
            s_val[tid] = (cur_ptr < my_k) ? my_val[cur_ptr] : -1e38;
            s_idx[tid] = (cur_ptr < my_k) ? my_idx[cur_ptr] : 0u;
        }
        barrier();
        memoryBarrierShared();
    }
}
