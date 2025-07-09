#pragma once

#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>
#include <Kokkos_Core.hpp>

namespace KB = KokkosBatched;

struct FunctorKernel_Matmul3D_Batched_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index k) const
    {
        auto A     = std::get<0>(views);
        auto X     = std::get<1>(views);
        auto X_out = std::get<2>(views);

        using ScalarType = typename decltype(X)::value_type;
        using ExeSpace   = typename Kokkos::DefaultHostExecutionSpace;

        const Index N = X.extent(0);
        const Index M = X.extent(1);
        const Index P = X_out.extent(0);
        const Index K = X.extent(2);

        if (k >= K)
            return;

        // get the slice for this K
        auto X_slice     = Kokkos::subview(X, Kokkos::ALL, Kokkos::ALL, k);
        auto X_out_slice = Kokkos::subview(X_out, Kokkos::ALL, Kokkos::ALL, k);

        KB::SerialGemm<KB::Trans::NoTranspose, KB::Trans::NoTranspose, KB::Algo::Gemm::Blocked>::
          invoke(1.0, A, X_slice, 0.0, X_out_slice);
    }
};


struct FunctorKernel_Matmul3D_Batched_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index k) const
    {
        auto A     = std::get<0>(views);
        auto X     = std::get<1>(views);
        auto X_out = std::get<2>(views);

        using ScalarType = typename decltype(X)::value_type;
        using ExeSpace   = typename Kokkos::DefaultExecutionSpace;

        const Index N = X.extent(0);
        const Index M = X.extent(1);
        const Index P = X_out.extent(0);
        const Index K = X.extent(2);

        if (k >= K)
            return;

        // get the slice for this K
        auto X_slice     = Kokkos::subview(X, Kokkos::ALL(), Kokkos::ALL(), k);
        auto X_out_slice = Kokkos::subview(X_out, Kokkos::ALL(), Kokkos::ALL(), k);

        KB::SerialGemm<KB::Trans::NoTranspose, KB::Trans::NoTranspose, KB::Algo::Gemm::Blocked>::
          invoke(1.0, A, X_slice, 0.0, X_out_slice);
    }
};



template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelMatMul3DBatched(KernelOptions& options, ParameterTypes&... data_views)
{
    constexpr int KernelRank = 1;

    auto name = "matrix-matrix multiplication across 3d z-slice";

    auto is_const = kernel_io_map(data_views...);
    auto views_   = pack(data_views...);
    auto views    = repack_views(views_);

    auto out        = std::get<2>(std::get<0>(views));
    unsigned long K = out.extent(2);

    RangeExtent<KernelRank> extent = range_extent(0, K);


    auto full_policy_collection =
      make_policy_from_hyperparameters<KernelRank, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(
        extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();

    return Kernel<KernelRank,
                  FunctorKernel_Matmul3D_Batched_Host,
                  FunctorKernel_Matmul3D_Batched_Device,
                  decltype(views),
                  decltype(is_const),
                  decltype(full_policy_collection),
                  decltype(policy_names)>(name,
                                          views,
                                          is_const,
                                          extent,
                                          options,
                                          full_policy_collection,
                                          policy_names);
}
