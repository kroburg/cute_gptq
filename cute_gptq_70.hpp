#pragma once

#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/tensor_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/tensor_predicate.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/tensor.hpp"

namespace cutlass::gemm {
  using namespace cute;

  static __device__ void
  ConvertPermutedInterleaved2(uint32_t val, __half2* __restrict__ out)
  {
    // immLut encodes binary operation F(a, b, c) we want to perform
    // for more details see https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3
    static constexpr uint32_t
            immLut = (0xf0 & 0xcc) | 0xaa;

    static constexpr uint32_t
            BOTTOM_MASK = 0x000f000f;
    static constexpr uint32_t
            TOP_MASK = 0x00f000f0;
    static constexpr uint32_t
            I4s_TO_F16s_MAGIC_NUM = 0x64006400;
    //static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    const __half2 zero64 = __halves2half2(64.f, 64.f);
    const __half2 zero1024 = __halves2half2(1024.f, 1024.f);
    const __half2 oneSixteenth = __halves2half2(0.0625f, 0.0625f);

    int* outInt = reinterpret_cast<int*>(out);

    const uint32_t topVal = val >> 8;
    // Extract elt_04 - (val & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
            : "=r"(outInt[0])
            : "r"(val), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // Extract elt_15 (val & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
            : "=r"(outInt[1])
            : "r"(val), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_26 (topVal & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
            : "=r"(outInt[2])
            : "r"(topVal), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // Extract elt_37 (topVal & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
            : "=r"(outInt[3])
            : "r"(topVal), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    out[0] = out[0] - zero1024;
    out[1] = out[1] * oneSixteenth - zero64;
    out[2] = out[2] - zero1024;
    out[3] = out[3] * oneSixteenth - zero64;
  }

  template <class EngineIn, class EngineOut, class TensorLayout, int N = cosize_v<TensorLayout>>
  static __device__
  void unpack_4bit(const Tensor<EngineIn, TensorLayout>& in, Tensor<EngineOut, TensorLayout>& out)
  {
    static_assert(is_rmem<EngineIn>::value, "Input tensor for conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value, "Output tensor for conversion must come from registers");
    static_assert(!(N % 8), "Must have multiple of 4 elements for fast F16 convert");
    //printf("%s", in.data());
    const uint32_t* source_ptr = reinterpret_cast<const uint32_t*>(in.engine().data());
    uint128_t* result_ptr = reinterpret_cast<uint128_t*>(out.data());
    for (int ii = 0; ii < N / 8; ++ii)
    {
      ConvertPermutedInterleaved2(*source_ptr, (__half2*)result_ptr);
      ++source_ptr;
      ++result_ptr;
    }
  }

  struct ZeroScale
  {
    half_t Zero;
    half_t Scale;
  };

  template <class EngineIn, class EngineZeroScale, class EngineOut, class TensorLayout, int N = cosize_v<TensorLayout>>
  static __device__
  void rescale(Tensor<EngineIn, TensorLayout>& in, Tensor<EngineZeroScale, TensorLayout>& zeroScale, Tensor<EngineOut, TensorLayout>& out)
  {
    if (false)
    if (cute::thread0())
    {
      printf("in:\n");
      print(in);
      print_tensor(in);
    }

    cute::transform(in, zeroScale, out, [](auto in, ZeroScale r) {
      auto result = r.Scale * (in - r.Zero);
      if (false)
      if (cute::thread0())
        printf("thread0 int4 value: %f, zero: %f, scale: %f, result: %f\n", (float)in, (float)r.Zero, (float)r.Scale, (float)result);
      return result;
    });

    if (false)
    if (cute::thread0())
    {
      auto zeroes = make_tensor_like(in);
      auto scales = make_tensor_like(in);
      cute::transform(zeroScale, zeroes, [](auto r) {
        return r.Zero;
      });
      cute::transform(zeroScale, scales, [](auto r) {
        return r.Scale;
      });
      printf("zeroes:\n");
      print_tensor(zeroes);
      printf("scales:\n");
      print_tensor(scales);
      printf("out:\n");
      print(out);
      print_tensor(out);
    }
  }

  struct KernelMultistageGptq { };

  struct MainloopSm70TwoStageGptq
  {
    constexpr static int Stages = 2;
    using ArchTag = arch::Sm70;
    using Schedule = KernelMultistageGptq;
    using ClusterShape = Shape<_1, _1, _1>;
  };
}

namespace cutlass::gemm::collective {
using namespace cute;
  template <
          class Dispatch_,
          class TileShape_,
          class ElementA_,
          class StrideA_,
          class ElementB_,
          class StrideB_,
          class TiledMma_,
          class GmemTiledCopyA_,
          class SmemLayoutAtomA_,
          class SmemCopyAtomA_,
          class TransformA_,
          class GmemTiledCopyB_,
          class SmemLayoutAtomB_,
          class SmemCopyAtomB_,
          class TransformB_,
          class GmemTiledCopyZS_>
  struct CollectiveMmaMixed;
template <
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_,
  class GmemTiledCopyZS_>
struct CollectiveMmaMixed<
    MainloopSm70TwoStageGptq,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_,
    GmemTiledCopyZS_>
{
  //
  // Type Aliases
  //
  using DispatchPolicy = MainloopSm70TwoStageGptq;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyZS = GmemTiledCopyZS_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}))));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}))));

  struct SharedStorage
  {
    cute::array_aligned<ElementA, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<half_t, cute::cosize_v<SmemLayoutB>> smem_b;
  };

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    const ZeroScale* ptr_BZS;
  };

  // Device side kernel params
  using Params = Arguments;

  //
  // Methods
  //

  CollectiveMmaMixed() = default;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& _, Arguments const& args, void* workspace) {
    (void) workspace;
    return args;
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  template <
    class FrgTensorD,
    class TensorA,
    class TensorB, class TensorBZS,
    class FrgTensorC,
    class KTileIterator,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator() (
      FrgTensorD &accum,
      TensorA gA,
      TensorB gB, TensorBZS gBZS,
      FrgTensorC const &src_accum,
      KTileIterator k_tile_iter, int k_tile_count,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf) 
  {
    using namespace cute;

    static_assert(is_rmem<FrgTensorD>::value, "D tensor must be rmem resident.");
    static_assert(is_gmem<TensorA>::value, "A tensor must be gmem resident.");
    static_assert(is_gmem<TensorB>::value, "B tensor must be gmem resident.");
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 2,
      "MainloopTwoStage must not have a smem shape with a pipeline mode.");
    static_assert(rank(SmemLayoutB{}) == 2,
      "MainloopTwoStage must not have a smem shape with a pipeline mode.");

    // Construct shared memory tiles
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    Tensor sA = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(storage.smem_b.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

    // Shift tensor so residue_k is at origin (Can't read any k_coord < residue_k)
    // This aligns the tensor with BLK_K for all but the 0th k_tile
    gA.data() = &gA(0, get<2>(residue_mnk), 0);
    gB.data() += gB.layout()(make_coord(0, get<2>(residue_mnk), 0));
    gBZS.data() = &gBZS(0, get<2>(residue_mnk), 0);

    // Partition the copying of A and B tiles across the threads
    GmemTiledCopyA gmem_tiled_copy_a;
    GmemTiledCopyB gmem_tiled_copy_b;
    GmemTiledCopyZS gmem_tiled_copy_zs;
    auto gmem_thr_copy_a = gmem_tiled_copy_a.get_slice(thread_idx);
    auto gmem_thr_copy_b = gmem_tiled_copy_b.get_slice(thread_idx);
    auto gmem_thr_copy_zs = gmem_tiled_copy_zs.get_slice(thread_idx);

    if (false)
    if (cute::thread0())
    {
      printf("gmem_tiled_copy_b:\n");
      print(gmem_tiled_copy_b);
      printf("\n");
      printf("gmem_tiled_copy_zs:\n");
      print(gmem_tiled_copy_zs);
      printf("\n");
    }

    Tensor tAgA = gmem_thr_copy_a.partition_S(gA);                             // (ACPY,ACPY_M,ACPY_K,k)
    Tensor tAsA = gmem_thr_copy_a.partition_D(sA);                             // (ACPY,ACPY_M,ACPY_K,PIPE)

    Tensor tBgB = gmem_thr_copy_b.partition_S(gB);                             // (BCPY,BCPY_N,BCPY_K,k)
    Tensor tBZSgB = gmem_thr_copy_zs.partition_S(gBZS);                             // (BCPY,BCPY_N,BCPY_K,k)

    Tensor tBsB = gmem_thr_copy_b.partition_D(sB);                             // (BCPY,BCPY_N,BCPY_K,PIPE)

    if (false)
    if (cute::thread0())
    {
      printf("tBgB:\n");
      print(tBgB);
      printf("\n");
      printf("tBZSgB:\n");
      print(tBZSgB);
      printf("\n");

      printf("\n");
    }

    // Allocate the register tiles for double buffering -- same shape as partitioned data
    Tensor tArA = make_fragment_like(tAsA);                                    // (ACPY,ACPY_M,ACPY_K)
    Tensor tBrB_load = make_fragment_like<int4_t>(tBsB);                                    // (BCPY,BCPY_N,BCPY_K)
    Tensor tBrB_decode = make_fragment_like<half_t>(tBsB);                                    // (BCPY,BCPY_N,BCPY_K)
    Tensor tBrBZS = make_fragment_like<ZeroScale>(tBsB);                                    // (BCPY,BCPY_N,BCPY_K)

    if (false)
    if (cute::thread0())
    {
      printf("tBsB:\n");
      print(tBsB);
      printf("\n");
      printf("tBrB_load:\n");
      print(tBrB_load);
      printf("\n");
      printf("tBrB_decode:\n");
      print(tBrB_decode);
      printf("\n");
    }

    //
    // PREDICATES
    //

    // Allocate predicate tensors for m and n
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)), Stride<_1,_0>{});
    Tensor tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)), Stride<_1,_0>{});

    // Construct identity layout for sA and sB
    Tensor cA = make_identity_tensor(make_shape(size<0>(sA), size<1>(sA)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cB = make_identity_tensor(make_shape(size<0>(sB), size<1>(sB)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tAcA = gmem_thr_copy_a.partition_S(cA);                             // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tBcB = gmem_thr_copy_b.partition_S(cB);                             // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    if (false)
    if (cute::thread0())
    {
      printf("tBcB:\n");
      print(tBcB);
      printf("\n");
    }

    // Set predicates for m bounds
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(tApA); ++m) {
      tApA(m,0) = get<0>(tAcA(0,m,0)) < get<0>(residue_mnk);  // blk_m coord < residue_m
    }
    // Set predicates for n bounds
    CUTLASS_PRAGMA_UNROLL
    for (int n = 0; n < size<0>(tBpB); ++n) {
      tBpB(n,0) = get<0>(tBcB(0,n,0)) < get<1>(residue_mnk);  // blk_n coord < residue_n
    }

    //
    // PREFETCH
    //

    // Clear the rmem tiles to account for predicated off loads
    clear(tArA);
    clear(tBrB_load);

    // Start async loads for 0th k-tile, where we take care of the k residue
    {
      Tensor tAgAk = tAgA(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tArA); ++k) {
        if (get<1>(tAcA(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gA shifted)
          copy_if(gmem_tiled_copy_a, tApA(_,k), tAgAk(_,_,k), tArA(_,_,k));
        }
      }
      Tensor tBgBk = tBgB(_,_,_,*k_tile_iter);
      Tensor tBgBZSk = tBZSgB(_,_,_,*k_tile_iter);
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<2>(tBrB_load); ++k) {
        if (get<1>(tBcB(0,0,k)) >= -get<2>(residue_mnk)) {      // blk_k coord < residue_k (gB shifted)
          {
            if (false)
            if (cute::thread0())
            {
              printf("k=%d tBgBk:\n", k);
              print(tBgBk(_, _, k));
              printf("\n");
              printf("k=%d tBgBZSk:\n", k);
              print(tBgBZSk(_, _, k));
              printf("\n");
              printf("k=%d tBrBk:\n", k);
              print(tBrB_load(_, _, k));
              printf("\n");
              printf("\n");
            }
          }
          copy_if(gmem_tiled_copy_b, tBpB(_,k), tBgBk(_,_,k), tBrB_load(_,_,k));
          copy_if(gmem_tiled_copy_zs, tBpB(_,k), tBgBZSk(_,_,k), tBrBZS(_,_,k));
        }
      }
      ++k_tile_iter;
      --k_tile_count;
    }

    // Tile MMA compute thread partitions and allocate accumulators
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCrA  = thr_mma.make_fragment_A(thr_mma.partition_A(sA));           // (MMA,MMA_M,MMA_K)
    Tensor tCrB  = thr_mma.make_fragment_B(thr_mma.partition_B(sB));           // (MMA,MMA_M,MMA_K)

    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(accum));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(src_accum));                 // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(accum));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(src_accum));                 // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                      // MMA_K

    //
    // Copy Atom retiling
    //

    auto thr_copy_A       = make_tiled_copy_A(SmemCopyAtomA{}, tiled_mma).get_thread_slice(thread_idx);
    Tensor tCsA           = thr_copy_A.partition_S(sA);
    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M

    auto thr_copy_B       = make_tiled_copy_B(SmemCopyAtomB{}, tiled_mma).get_thread_slice(thread_idx);
    Tensor tCsB           = thr_copy_B.partition_S(sB);
    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N

    //
    // Prologue
    //

    // Copy rmem to smem
    copy(tArA, tAsA);
    unpack_4bit(tBrB_load, tBrB_decode);

    // @todo Out into tBsB?
    rescale(tBrB_decode, tBrBZS, tBrB_decode);

    copy(tBrB_decode, tBsB);
    // Clear accumulators
    __syncthreads();

    // Load A, B smem->rmem for k=0
    copy(tCsA(_,_,0), tCrA_copy_view(_,_,0));
    copy(tCsB(_,_,0), tCrB_copy_view(_,_,0));
    //
    // Mainloop
    //

    // Size of the k-tiles's outer product mode (k)
    auto K_BLOCK_MAX = size<2>(tCrA);

    CUTLASS_PRAGMA_NO_UNROLL
    while (k_tile_count > -1)
    {
      // Pipeline the outer products with a static for loop
      for_each(make_int_sequence<K_BLOCK_MAX>{}, [&] (auto k_block) 
      {
        if (k_block == K_BLOCK_MAX - 1) 
        {
          __syncthreads();

          // Copy rmem to smem
          copy(tArA, tAsA);

          unpack_4bit(tBrB_load, tBrB_decode);
          // @todo Out into tBsB?
          rescale(tBrB_decode, tBrBZS, tBrB_decode);

          copy(tBrB_decode, tBsB);
          __syncthreads();
        }

        // Load A, B smem->rmem for k+1
        int k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;    // static
        copy(tCsA(_,_,k_block_next), tCrA_copy_view(_,_,k_block_next));
        copy(tCsB(_,_,k_block_next), tCrB_copy_view(_,_,k_block_next));
        if (k_block == 0) 
        {
          if (k_tile_count <= 0) {
            clear(tApA);
            clear(tBpB);
          }
          copy_if(gmem_tiled_copy_a, tApA, tAgA(_,_,_,*k_tile_iter), tArA);
          copy_if(gmem_tiled_copy_b, tBpB, tBgB(_,_,_,*k_tile_iter), tBrB_load);
          copy_if(gmem_tiled_copy_zs, tBpB, tBZSgB(_,_,_,*k_tile_iter), tBrBZS);
          ++k_tile_iter;
          --k_tile_count;
        }

        // transform before compute
        cute::transform(tCrA(_,_,k_block), TransformA{});
        cute::transform(tCrB(_,_,k_block), TransformB{});

        // Thread-level register gemm for k
        // disambiguate gemm (shared with the namespace name)
        cute::gemm(tiled_mma, accum, tCrA(_,_,k_block), tCrB(_,_,k_block), src_accum);
      });
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////


namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class GemmUniversal<
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<KernelMultistageGptq, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  static constexpr int GroupSize = 128;
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;

  static_assert(rank(ProblemShape{}) == 3 or rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  static_assert(cute::is_void_v<TileScheduler_> or cute::is_same_v<TileScheduler_, PersistentScheduler>,
    "SM70 kernel does not support specializing the tile scheduler.");
  using TileScheduleTag = TileScheduler_;
  using TileScheduler = typename detail::TileSchedulerSelector<
    TileScheduler_, ArchTag, TileShape,
    cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;
  static_assert(cute::is_same_v<ElementAccumulator, typename CollectiveEpilogue::ElementAccumulator>,
    "Mainloop and epilogue do not agree on accumulator value type.");

  // MSVC requires the cast to fix a warning-as-error.
  static constexpr int SharedStorageSize = static_cast<int>(cute::max(
      sizeof(typename CollectiveMainloop::SharedStorage),
      sizeof(typename CollectiveEpilogue::SharedStorage)));

  static constexpr uint32_t MaxThreadsPerBlock = cute::size(TiledMma{});
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode;
    ProblemShape problem_shape;
    MainloopParams mainloop;
    EpilogueParams epilogue;
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    (void) workspace;
    return {
      args.mode,
      args.problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, args.mainloop, workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, workspace)
    };
  }

  static bool
  can_implement(Arguments const& args) {
    return args.mode == GemmUniversalMode::kGemm or
          (args.mode == GemmUniversalMode::kBatched && rank(ProblemShape{}) == 4);
  }

  static int
  get_workspace_size(Arguments const& args) {
    return 0;
  }

  static
  cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr) {
    return Status::kSuccess;
  }

  static dim3
  get_grid_shape(Params const& params) {
    int batch_count = 1;
    if constexpr (rank(ProblemShape{}) == 4) {
      batch_count = cute::size<3>(params.problem_shape);
    }

    return dim3(
      cute::size(cute::ceil_div(cute::shape<0>(params.problem_shape), cute::shape<0>(TileShape{}))),
      cute::size(cute::ceil_div(cute::shape<1>(params.problem_shape), cute::shape<1>(TileShape{}))),
      batch_count
    );
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;

    // Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    int thread_idx = int(threadIdx.x);
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)
    auto [m_coord, n_coord, l_coord] = blockIdx;
    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);                                        // (m,n,k,l)

    // Represent the full tensors
    Tensor mA_mkl = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,L), params.mainloop.dA); //(m,k,l)
    Tensor mB_nkl = make_tensor(make_subbyte_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,L), params.mainloop.dB); //(n,k,l)

    auto BZS_shape = make_shape(N, make_shape(Int<GroupSize>{}, K / GroupSize), L);
    auto BZS_stride = make_stride(size<0>(params.mainloop.dB) / GroupSize, Stride<_0, _1>{}, _0{});
    //auto BZS_shape = make_shape(N, make_shape(K / GroupSize, Int<GroupSize>{}), L);
    //auto BZS_stride = make_stride(size<0>(params.mainloop.dB) / GroupSize, Stride<_1, _0>{}, _0{});
    auto BZS_ptr = params.mainloop.ptr_BZS;
    Tensor mBZS_nkl = make_tensor(BZS_ptr, BZS_shape, BZS_stride);

    if (false)
    if (cute::thread0())
    {
      printf("mB_nkl:\n");
      print(mB_nkl);
      printf("\n");
      printf("mBZS_nkl:\n");
      print(mBZS_nkl);
      printf("\n");
    }

    // Get batch slice
    Tensor mA_mk = mA_mkl(_,_,l_coord);                                                                        // (m,k)
    Tensor mB_nk = mB_nkl(_,_,l_coord);                                                                        // (n,k)
    Tensor mBZS_nk = mBZS_nkl(_,_,l_coord);                                                                      // (n,k)

    if (false)
    if (cute::thread0())
    {
      printf("blk_coord_mnkl:\n");
      print(blk_coord_mnkl);
      printf("\n");
    }

    if (false)
    if (cute::thread0())
    {
      printf("mB_nk:\n");
      print(mB_nk);
      printf("\n");
      printf("mBZS_nk:\n");
      print(mBZS_nk);
      printf("\n");
      printf("\n");
    }

    // Slice to get the tiles this thread block is responsible for
    Tensor gA = local_tile(mA_mk, blk_shape, take<0,3>(blk_coord_mnkl), Step<_1, X,_1>{});           // (BLK_M,BLK_K,k)
    Tensor gB = local_tile(mB_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});           // (BLK_N,BLK_K,k)
    Tensor gBZS = local_tile(mBZS_nk, blk_shape, take<0,3>(blk_coord_mnkl), Step< X,_1,_1>{});         // (BLK_N,BLK_K,k)

    if (false)
    if (cute::thread0())
    {
      printf("gB:\n");
      print(gB);
      printf("\n");
      printf("gBZS:\n");
      print(gBZS);
      printf("\n");
     }

    // Compute tile residues for predication
    auto m_max_coord = M - size<0>(gA) * get<0>(blk_coord_mnkl);                             // M - BLK_M * m_coord
    auto n_max_coord = N - size<0>(gB) * get<1>(blk_coord_mnkl);                             // N - BLK_N * n_coord
    auto k_residue   = K - size<1>(gA) * size<2>(gA);                                        // K - BLK_K * k_coord_max
    auto residue_mnk = make_tuple(m_max_coord, n_max_coord, k_residue);

    // Allocate the tiled_mma and the accumulators for the (M,N) blk_shape
    TiledMma tiled_mma;
    Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape)); // (MMA,MMA_M,MMA_N)
    clear(accumulators);

    auto k_tile_iter  = cute::make_coord_iterator(shape<2>(gA));
    int  k_tile_count = size<2>(gA);

    // Perform the collective scoped MMA
    CollectiveMainloop collective_mma;
    collective_mma(
      accumulators,
      gA,
      gB, gBZS,
      accumulators,
      k_tile_iter, k_tile_count,
      residue_mnk,
      thread_idx,
      smem_buf
    );

    // Epilogue and write to gD
    CollectiveEpilogue epilogue{params.epilogue};
    epilogue(
      problem_shape_MNKL,
      blk_shape,
      blk_coord_mnkl,
      accumulators,
      tiled_mma,
      residue_mnk,
      thread_idx,
      smem_buf
    );
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
