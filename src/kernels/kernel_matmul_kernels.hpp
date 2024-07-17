#pragma once

#include "KokkosBlas3_gemm.hpp"
#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"

struct FunctorKernel_MatMul_Kernels_Host
{
    // so, this one is the host and it doesn't do CUBLAS, it just uses regular blas I think
    template<typename ViewsTuple>
    KOKKOS_FUNCTION void operator()(ViewsTuple views) const
    {
        auto A = std::get<0>(views);
        auto B = std::get<1>(views);
        auto C = std::get<2>(views);

        // C is our output, and we need to get our values from a and b
        const double alpha = double(1.0);
        const double beta  = double(0.0);

        // does KokkosBlas support CPU?
        KokkosBlas::gemm("N", "N", alpha, A, B, beta, C);

        // and that should be it?
    }
};

struct FunctorKernel_MatMul_Kernels_Device
{
    template<typename ViewsTuple>
    KOKKOS_FUNCTION void operator()(ViewsTuple views) const
    {
        auto A = std::get<0>(views);
        auto B = std::get<1>(views);
        auto C = std::get<2>(views);

        // C is our output, and we need to get our values from a and b
        const double alpha = double(1.0);
        const double beta  = double(0.0);

        // Kokkos BLAS, both should be "N" and not transpose
        KokkosBlas::gemm("N", "N", alpha, A, B, beta, C);
    }
};

#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))

template<typename... ParameterTypes>
inline auto KernelMatMulKokkosKernel(KernelOptions& options, ParameterTypes&... data_views)
{
    auto name = "matrix-matrix multiply (kokkos kernel)";

    auto is_const = kernel_io_map(data_views...);

    auto views_ = pack(data_views...);

    auto views = repack_views(views_);

    auto out    = get_v(0, 2, views);
    auto extent = range_extent();


    // then build up the policy collection
    auto full_policy_collection = create_range_policy_device<0>(extent);
    auto policy_names           = create_range_policy_device_collection();
    // NOTE: this is a kernelrank 0


    return Kernel<0,
                  FunctorKernel_MatMul_Kernels_Host,
                  FunctorKernel_MatMul_Kernels_Device,
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