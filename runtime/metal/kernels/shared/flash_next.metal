#include "metal/abi/FlashNext.h"
#include "metal/abi/KernelABI.h"
template <class W>
inline void flash_group_rms(device const bfloat *x, device const W *w,
                            device bfloat *y, constant FlashNextParams &p,
                            uint group, uint tid, uint lane, uint sg,
                            threadgroup float *partial) {
  const ulong offset = ulong(group) * p.width;
  float sum = 0;
  for (uint c = tid; c < p.width; c += 256) {
    float v = float(x[offset + c]);
    sum += v * v;
  }
  sum = simd_sum(sum);
  if (lane == 0)
    partial[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float s = 0;
    for (uint i = 0; i < 8; ++i)
      s += partial[i];
    partial[0] = rsqrt(s / p.width + p.epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint c = tid; c < p.width; c += 256)
    y[offset + c] = bfloat(float(x[offset + c]) * partial[0] *
                           float(w[(group % p.streams) * p.width + c]));
}
kernel void flash_next_group_rms(device const bfloat *x [[buffer(0)]],
                                 device const bfloat *w [[buffer(1)]],
                                 device bfloat *y [[buffer(2)]],
                                 constant FlashNextParams &p [[buffer(3)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  flash_group_rms(x, w, y, p, group, tid, lane, sg, partial);
}
kernel void flash_next_group_rms_f32(
    device const bfloat *x [[buffer(0)]], device const float *w [[buffer(1)]],
    device bfloat *y [[buffer(2)]], constant FlashNextParams &p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  flash_group_rms(x, w, y, p, group, tid, lane, sg, partial);
}
inline float flash_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
kernel void flash_next_hc_silu(device const bfloat *x [[buffer(0)]],
                               device bfloat *y [[buffer(1)]],
                               constant FlashNextParams &p [[buffer(2)]],
                               uint i [[thread_position_in_grid]]) {
  if (i >= p.width * p.rows)
    return;
  const float v = float(bfloat(float(x[i]) / p.streams));
  y[i] = bfloat(v * flash_sigmoid(v));
}
kernel void flash_next_hc_mix(device const bfloat *x [[buffer(0)]],
                              device const bfloat *gate [[buffer(1)]],
                              device bfloat *y [[buffer(2)]],
                              constant FlashNextParams &p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
  if (i >= p.width * p.rows)
    return;
  const ulong base = ulong(i / p.width) * p.width * p.streams + i % p.width;
  float sum = 0;
  for (uint s = 0; s < p.streams; ++s) {
    const ulong j = base + ulong(s) * p.width;
    const bfloat sigmoid = bfloat(flash_sigmoid(float(gate[j])));
    sum += float(bfloat(float(x[j]) * float(sigmoid)));
  }
  y[i] = bfloat(sum / p.streams);
}
kernel void flash_next_hc_scatter(device const bfloat *x [[buffer(0)]],
                                  device const bfloat *block [[buffer(1)]],
                                  device const bfloat *inject [[buffer(2)]],
                                  device bfloat *y [[buffer(3)]],
                                  constant FlashNextParams &p [[buffer(4)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i >= p.width * p.streams * p.rows)
    return;
  const uint row = i / (p.width * p.streams),
             stream = (i / p.width) % p.streams;
  const bfloat sigmoid = bfloat(flash_sigmoid(
      float(bfloat(float(inject[row * p.streams + stream]) / p.streams))));
  const bfloat weight = bfloat(2 * float(sigmoid));
  const bfloat delta =
      bfloat(float(block[row * p.width + i % p.width]) * float(weight));
  y[i] = bfloat(float(x[i]) + float(delta));
}
kernel void flash_next_ple_gate(device const bfloat *key [[buffer(0)]],
                                device const bfloat *query [[buffer(1)]],
                                device const bfloat *value [[buffer(2)]],
                                device bfloat *out [[buffer(3)]],
                                constant FlashNextParams &p [[buffer(4)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float partial[8];
  const ulong off = ulong(group) * p.width;
  float dot = 0;
  for (uint c = tid; c < p.width; c += 256)
    dot += float(bfloat(float(key[off + c]) * float(query[off + c])));
  dot = simd_sum(dot);
  if (lane == 0)
    partial[sg] = dot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float sum = 0;
    for (uint i = 0; i < 8; ++i)
      sum += partial[i];
    const float v = float(bfloat(float(bfloat(sum)) / sqrt(float(p.width))));
    const float root = float(bfloat(sqrt(float(bfloat(max(abs(v), 1e-6f))))));
    const float sign = v > 0 ? 1.0f : v < 0 ? -1.0f : 0.0f;
    partial[0] = float(bfloat(flash_sigmoid(root * sign)));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const ulong voff = ulong(group / p.streams) * p.width;
  for (uint c = tid; c < p.width; c += 256)
    out[off + c] = bfloat(float(value[voff + c]) * partial[0]);
}
kernel void flash_next_ple_conv(device const bfloat *direct [[buffer(0)]],
                                device const bfloat *norm [[buffer(1)]],
                                device const bfloat *state [[buffer(2)]],
                                device const float *w [[buffer(3)]],
                                device bfloat *out [[buffer(4)]],
                                constant FlashNextConvParams &p [[buffer(5)]],
                                uint i [[thread_position_in_grid]]) {
  const uint channels = p.width * p.streams;
  if (i >= channels * p.rows)
    return;
  const uint row = i / channels, ch = i % channels,
             history = (p.taps - 1) * p.dilation;
  float sum = 0;
  for (uint tap = 0; tap < p.taps; ++tap) {
    const uint source = row + tap * p.dilation;
    const float x = source < history
                        ? float(state[ulong(source) * channels + ch])
                        : float(norm[ulong(source - history) * channels + ch]);
    sum += x * w[ulong(ch) * p.taps + tap];
  }
  const float z = float(bfloat(sum));
  const bfloat activation = bfloat(z * flash_sigmoid(z));
  out[i] = bfloat(float(direct[i]) + float(activation));
}
kernel void flash_next_ple_commit(device const bfloat *norm [[buffer(0)]],
                                  device const bfloat *state [[buffer(1)]],
                                  device bfloat *next [[buffer(2)]],
                                  constant FlashNextConvParams &p [[buffer(3)]],
                                  uint ch [[thread_position_in_grid]]) {
  const uint channels = p.width * p.streams,
             history = (p.taps - 1) * p.dilation;
  if (ch >= channels)
    return;
  // Ascending rows read ahead of writes even when old/new state alias.
  for (uint r = 0; r < history; ++r) {
    const uint source = p.rows + r;
    next[ulong(r) * channels + ch] =
        source < history ? state[ulong(source) * channels + ch]
                         : norm[ulong(source - history) * channels + ch];
  }
}
kernel void flash_next_qsa_pool(device const bfloat *raw [[buffer(0)]],
                                device bfloat *pooled [[buffer(1)]],
                                constant FlashNextPoolParams &p [[buffer(2)]],
                                uint i [[thread_position_in_grid]]) {
  if (i >= p.width * p.blocks)
    return;
  const uint block = p.first_block + i / p.width, col = i % p.width;
  float sum = 0;
  for (uint t = 0; t < p.ratio; ++t)
    sum += float(raw[(ulong(block) * p.ratio + t) * p.width + col]);
  pooled[ulong(block) * p.width + col] = bfloat(sum / p.ratio);
}
kernel void flash_next_rope(device const bfloat *input [[buffer(0)]],
                            device const float *cosine [[buffer(1)]],
                            device const float *sine [[buffer(2)]],
                            device const uint *positions [[buffer(3)]],
                            device bfloat *output [[buffer(4)]],
                            constant FlashNextRopeParams &p [[buffer(5)]],
                            uint i [[thread_position_in_grid]]) {
  const uint pairs = p.rotary_dimension / 2, tasks = p.width - pairs;
  if (i >= tasks * p.heads * p.rows)
    return;
  const uint headrow = i / tasks, col = i % tasks;
  const ulong base = ulong(headrow) * p.width;
  if (col >= pairs) {
    output[base + col + pairs] = input[base + col + pairs];
    return;
  }
  const uint pos = positions[headrow / p.heads];
  const ulong angle = ulong(pos) * pairs + col;
  if (pos >= p.table_rows) {
    output[base + col] = bfloat(NAN);
    output[base + col + pairs] = bfloat(NAN);
    return;
  }
  const float a = float(input[base + col]),
              b = float(input[base + col + pairs]), c = cosine[angle],
              s = sine[angle];
  output[base + col] = bfloat(a * c - b * s);
  output[base + col + pairs] = bfloat(b * c + a * s);
}
kernel void flash_next_qsa_scores(device const bfloat *query [[buffer(0)]],
                                  device const bfloat *keys [[buffer(1)]],
                                  device const uint *positions [[buffer(2)]],
                                  device float *scores [[buffer(3)]],
                                  constant FlashNextQsaParams &p [[buffer(4)]],
                                  uint i [[thread_position_in_grid]]) {
  if (i >= p.rows * p.blocks)
    return;
  const uint row = i / p.blocks, block = i % p.blocks;
  if ((ulong(block) + 1) * p.ratio > ulong(positions[row]) + 1) {
    scores[i] = -INFINITY;
    return;
  }
  float score = 0;
  for (uint h = 0; h < p.heads; ++h) {
    float dot = 0;
    for (uint c = 0; c < p.width; ++c)
      dot += float(query[(ulong(row) * p.heads + h) * p.width + c]) *
             float(keys[ulong(block) * p.width + c]);
    score += max(0.0f, dot);
  }
  scores[i] = score / sqrt(float(p.width));
}
// Worse values lead a min-heap; ties prefer the smaller block index.
inline bool flash_qsa_worse(int a, int b, device const float *scores) {
  return scores[a] < scores[b] || (scores[a] == scores[b] && a > b);
}
kernel void flash_next_qsa_select(device const float *all_scores [[buffer(0)]],
                                  device const uint *positions [[buffer(1)]],
                                  device int *all_selected [[buffer(2)]],
                                  constant FlashNextQsaParams &p [[buffer(3)]],
                                  uint row [[threadgroup_position_in_grid]]) {
  const uint stride = p.budget + p.ratio - 1;
  device int *ids = all_selected + ulong(row) * stride;
  device const float *scores = all_scores + ulong(row) * p.blocks;
  const uint visible = positions[row] + 1,
             complete = min(visible / p.ratio, p.blocks);
  const uint limit = min(complete, p.budget / p.ratio);
  uint count = 0;
  for (uint block = 0; block < complete; ++block) {
    if (count < limit) {
      uint at = count++;
      ids[at] = int(block);
      while (at) {
        const uint parent = (at - 1) / 2;
        if (!flash_qsa_worse(ids[at], ids[parent], scores))
          break;
        const int tmp = ids[at];
        ids[at] = ids[parent];
        ids[parent] = tmp;
        at = parent;
      }
    } else if (limit && flash_qsa_worse(ids[0], int(block), scores)) {
      ids[0] = int(block);
      uint at = 0;
      for (;;) {
        uint child = 2 * at + 1;
        if (child >= count)
          break;
        if (child + 1 < count &&
            flash_qsa_worse(ids[child + 1], ids[child], scores))
          ++child;
        if (!flash_qsa_worse(ids[child], ids[at], scores))
          break;
        const int tmp = ids[at];
        ids[at] = ids[child];
        ids[child] = tmp;
        at = child;
      }
    }
  }
  // Expand backwards because the temporary heap occupies the same buffer.
  for (uint reverse = count; reverse; --reverse) {
    const uint at = reverse - 1, block = uint(ids[at]);
    for (uint member = 0; member < p.ratio; ++member)
      ids[at * p.ratio + member] = int(block * p.ratio + member);
  }
  uint end = count * p.ratio;
  const uint tail = (visible / p.ratio) * p.ratio;
  for (uint t = tail; t < visible && end < stride; ++t)
    ids[end++] = int(t);
  for (; end < stride; ++end)
    ids[end] = -1;
}
#include "metal/kernels/common/kv_extent.h"
template <class W>
inline void flash_gdn_gate(device const bfloat *input,
                           device const bfloat *gate, device const W *weight,
                           device bfloat *output,
                           constant FlashNextGdnParams &p, uint task, uint tid,
                           uint lane, uint sg, threadgroup float *partial) {
  const uint token = task / p.value_heads, head = task % p.value_heads;
  const ulong base = ulong(task) * p.width;
  float sum = 0;
  for (uint c = tid; c < p.width; c += 256) {
    const float v = float(input[base + c]);
    sum += v * v;
  }
  sum = simd_sum(sum);
  if (lane == 0)
    partial[sg] = sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float s = 0;
    for (uint i = 0; i < 8; ++i)
      s += partial[i];
    partial[0] = rsqrt(s / p.width + 1e-6f);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint perkey = p.value_heads / p.key_heads;
  const uint outhead =
      p.tiled ? (head % perkey) * p.key_heads + head / perkey : head;
  const ulong outbase = (ulong(token) * p.value_heads + outhead) * p.width;
  for (uint c = tid; c < p.width; c += 256) {
    const bfloat normalized =
        bfloat(float(input[base + c]) * partial[0] * float(weight[c]));
    output[outbase + c] =
        bfloat(float(normalized) *
               flash_sigmoid(float(
                   gate[ulong(token) * p.gate_stride + head * p.width + c])));
  }
}
#define FLASH_GDN_GATE(Name, W)                                                \
  kernel void Name(device const bfloat *input [[buffer(0)]],                   \
                   device const bfloat *gate [[buffer(1)]],                    \
                   device const W *weight [[buffer(2)]],                       \
                   device bfloat *output [[buffer(3)]],                        \
                   constant FlashNextGdnParams &p [[buffer(4)]],               \
                   uint task [[threadgroup_position_in_grid]],                 \
                   uint tid [[thread_index_in_threadgroup]],                   \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint sg [[simdgroup_index_in_threadgroup]]) {               \
    threadgroup float partial[8];                                              \
    flash_gdn_gate(input, gate, weight, output, p, task, tid, lane, sg,        \
                   partial);                                                   \
  }
FLASH_GDN_GATE(flash_next_gdn_gate, bfloat)
FLASH_GDN_GATE(flash_next_gdn_gate_f32, float)

template <class Cache>
inline void
flash_selected_attention(device const bfloat *query, device const bfloat *gate,
                         device const SplashKvPage *table,
                         device const int *selected, device bfloat *output,
                         constant FlashNextAttentionParams &p, uint2 group,
                         uint tid, uint lane, uint sg,
                         threadgroup float *partial, threadgroup float *state) {
  const uint head = group.x, row = group.y, kvhead = head / (p.query_heads / 2);
  const ulong off = (ulong(row) * p.query_heads + head) * 256;
  const float q = float(query[off + tid]);
  float acc = 0;
  if (tid == 0) {
    state[0] = -INFINITY;
    state[3] = 0;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  SplashKvAddressing<2, Cache> address(p.kv, kvhead);
  for (uint slot = 0; slot < p.selection_stride; ++slot) {
    const int token = selected[ulong(row) * p.selection_stride + slot];
    if (token < 0)
      continue;
    const uint page = uint(token) / 32, local = uint(token) % 32;
    if (page >= p.page_table_entries || !table[page]) {
      acc = NAN;
      break;
    }
    const auto tensors = address.page(table[page]);
    float key = float(tensors.keys[local * 256 + tid]);
    float value = float(tensors.values[tid * 32 + local]);
    if constexpr (SplashKvPageBytes<2, Cache>::Quantized) {
      key *= tensors.key_scales[local];
      value *= tensors.value_scales[local];
    }
    const float dot = simd_sum(q * key);
    if (lane == 0)
      partial[sg] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
      float score = 0;
      for (uint i = 0; i < 8; ++i)
        score += partial[i];
      score *= 0.0625f;
      const float maximum = max(state[0], score);
      state[1] = exp(state[0] - maximum);
      state[2] = exp(score - maximum);
      state[3] = state[3] * state[1] + state[2];
      state[0] = maximum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    acc = acc * state[1] + value * state[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const bfloat attention = bfloat(state[3] > 0 ? acc / state[3] : acc);
  const bfloat activated = bfloat(flash_sigmoid(float(gate[off + tid])));
  output[off + tid] = bfloat(float(attention) * float(activated));
}
#define FLASH_SELECTED_ATTENTION(Name, Cache)                                  \
  kernel void Name(device const bfloat *query [[buffer(0)]],                   \
                   device const bfloat *gate [[buffer(1)]],                    \
                   device const SplashKvPage *table [[buffer(2)]],             \
                   device const int *selected [[buffer(3)]],                   \
                   device bfloat *output [[buffer(4)]],                        \
                   constant FlashNextAttentionParams &p [[buffer(5)]],         \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint tid [[thread_index_in_threadgroup]],                   \
                   uint lane [[thread_index_in_simdgroup]],                    \
                   uint sg [[simdgroup_index_in_threadgroup]]) {               \
    threadgroup float partial[8], state[4];                                    \
    flash_selected_attention<Cache>(query, gate, table, selected, output, p,   \
                                    group, tid, lane, sg, partial, state);     \
  }
FLASH_SELECTED_ATTENTION(flash_next_qsa_attention_bf16, bfloat)
FLASH_SELECTED_ATTENTION(flash_next_qsa_attention_int8, int8_t)

#define FLASH_COPY_ROWS(Name, Type) \
kernel void Name(device const Type *x [[buffer(0)]], device Type *out [[buffer(1)]], \
                 constant FlashNextRowsParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) { \
  if (ulong(i) >= ulong(p.storage_rows) * p.output_stride) return; \
  const uint row = i / p.output_stride, col = i % p.output_stride; \
  out[i] = row < p.valid_rows && col < p.width ? x[ulong(row) * p.input_stride + col] : Type(0); \
}
FLASH_COPY_ROWS(flash_next_copy_rows_bf16, bfloat)
FLASH_COPY_ROWS(flash_next_copy_rows_f32, float)

kernel void flash_next_broadcast(device const bfloat *x [[buffer(0)]],
                                 device bfloat *out [[buffer(1)]],
                                 constant FlashNextParams &p [[buffer(2)]],
                                 uint i [[thread_position_in_grid]]) {
  if (i < p.width * p.streams * p.rows)
    out[i] = x[ulong(i / (p.width * p.streams)) * p.width + i % p.width];
}
#define FLASH_BINARY(Name, Expression)                                         \
  kernel void Name(device const bfloat *a [[buffer(0)]],                       \
                   device const bfloat *b [[buffer(1)]],                       \
                   device bfloat *out [[buffer(2)]],                           \
                   constant FlashNextParams &p [[buffer(3)]],                  \
                   uint i [[thread_position_in_grid]]) {                       \
    if (i >= p.width * p.rows)                                                 \
      return;                                                                  \
    const float x = float(a[i]), y = float(b[i]);                              \
    out[i] = bfloat(Expression);                                               \
  }
FLASH_BINARY(flash_next_add, x + y)
FLASH_BINARY(flash_next_silu_mul, float(bfloat(x *flash_sigmoid(x))) * y)
FLASH_BINARY(flash_next_sigmoid_mul, x *float(bfloat(flash_sigmoid(y))))
kernel void flash_next_scaled_accum(device const bfloat *x [[buffer(0)]],
                                    device float *acc [[buffer(1)]],
                                    constant FlashNextParams &p [[buffer(2)]],
                                    uint i [[thread_position_in_grid]]) {
  if (i < p.width * p.rows)
    acc[i] += float(x[i]) * p.epsilon;
}
kernel void flash_next_float_to_bf16(device const float *x [[buffer(0)]],
                                     device bfloat *out [[buffer(1)]],
                                     constant FlashNextParams &p [[buffer(2)]],
                                     uint i [[thread_position_in_grid]]) {
  if (i < p.width * p.rows)
    out[i] = bfloat(x[i]);
}
