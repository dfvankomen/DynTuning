#pragma once

#include "Core"
#include "KokkosBlas3_gemm.hpp"
#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"

struct FunctorKernel_MatMul_Eigen_Host
{
    // so, this one is the host and it doesn't do CUBLAS, it just uses regular blas I think
    template<typename ViewsTuple>
    void operator()(ViewsTuple views) const
    {
        auto A = std::get<0>(views);
        auto B = std::get<1>(views);
        auto C = std::get<2>(views);

        // convert to eigen, then do eigen matrix multiplication

        Eigen::Map<DynMatrix2D> A_eigen(A.data(), A.extent(0), A.extent(1));
        Eigen::Map<DynMatrix2D> B_eigen(B.data(), B.extent(0), B.extent(1));
        Eigen::Map<DynMatrix2D> C_eigen(C.data(), C.extent(0), C.extent(1));

        // C is our output, and we need to get our values from a and b
        const double alpha = double(1.0);
        const double beta  = double(0.0);

        // matrix-matrix multiplication in Eigen is supported directly
        C_eigen = A_eigen * B_eigen;

        std::cout << A_eigen << std::endl;
        std::cout << B_eigen << std::endl;
        std::cout << C_eigen << std::endl;
    }
};

struct FunctorKernel_MatMul_Kokkos_Device
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
inline auto KernelMatMulEigenKokkosKernel(KernelOptions& options, ParameterTypes&... data_views)
{
    auto name = "matrix-matrix multiply (eigen cpu, kokkos kernel gpu)";

    auto is_const = kernel_io_map(data_views...);

    auto views_ = pack(data_views...);

    auto views = repack_views(views_);

    auto out    = get_v(0, 2, views);
    auto extent = range_extent();

    return Kernel<0,
                  FunctorKernel_MatMul_Eigen_Host,
                  FunctorKernel_MatMul_Kokkos_Device,
                  decltype(views),
                  decltype(is_const)>(name, views, is_const, extent, options);
}