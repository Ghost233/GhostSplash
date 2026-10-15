// Isolated TensorOps operand-format probe. No quantizer, rotation or softmax.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
using namespace metal;
using namespace mpp::tensor_ops;

struct KvTensorParams {
  uint pages;
  uint page_bytes;
};

template <uint M, uint N, uint K, class Element, class Storage>
inline void kv_tensor_matmul(device bfloat *input, device Storage *packed,
                             device float *output,
                             constant KvTensorParams &params, uint task) {
  constexpr uint PhysicalM = (M + 7) / 8 * 8;
  auto a = tensor(input, dextents<int, 2>{K, PhysicalM}, array<int, 2>{1, K});
  auto a0 = a.template slice<K, PhysicalM>(0, 0);
  auto first = tensor<device Element, dextents<int, 2>, tensor_inline>(
      packed, dextents<int, 2>{K, N}, array<int, 2>{1, K});
  auto b0 = first.template slice<K, N>(0, 0);
  constexpr auto descriptor = matmul2d_descriptor(
      PhysicalM, N, K, false, true, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<descriptor, execution_simdgroups<8>> op;
  auto acc = op.template get_destination_cooperative_tensor<
      decltype(a0), decltype(b0), float>();
  for (ushort i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = 0.0f;
  // Accumulating independent pages isolates matmul/read costs; it is not
  // attention. A group writes once so output traffic does not hide KV reads.
  for (uint page = 0; page < params.pages; ++page) {
    ulong offset = (ulong(task) * params.pages + page) * params.page_bytes;
    auto b = tensor<device Element, dextents<int, 2>, tensor_inline>(
        packed + offset / sizeof(Storage), dextents<int, 2>{K, N},
        array<int, 2>{1, K});
    auto slice = b.template slice<K, N>(0, 0);
    op.run(a0, slice, acc);
  }
  auto c = tensor(output + ulong(task) * PhysicalM * N, dextents<int, 2>{N, PhysicalM},
                  array<int, 2>{1, N});
  acc.store(c.template slice<N, PhysicalM>(0, 0));
}

#define KV_MM(Name, M, N, K, Element, Storage) \
  kernel void Name(device bfloat *input [[buffer(0)]], \
                   device Storage *packed [[buffer(1)]], \
                   device float *output [[buffer(2)]], \
                   constant KvTensorParams &params [[buffer(3)]], \
                   uint task [[threadgroup_position_in_grid]]) { \
    kv_tensor_matmul<M, N, K, Element>(input, packed, output, params, task); \
  }
#define KV_FORMATS(Op, M, N, K) \
  KV_MM(Op##_m##M##_bf16, M, N, K, bfloat, bfloat) \
  KV_MM(Op##_m##M##_int8, M, N, K, int8_t, int8_t) \
  KV_MM(Op##_m##M##_int4, M, N, K, int4b_format, uchar) \
  KV_MM(Op##_m##M##_int2, M, N, K, int2b_format, uchar)
KV_FORMATS(qk, 4, 32, 256)
KV_FORMATS(qk, 32, 32, 256)
KV_FORMATS(pv, 4, 256, 32)
KV_FORMATS(pv, 32, 256, 32)
