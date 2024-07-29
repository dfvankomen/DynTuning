
#pragma once

#include "Eigen"
#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"



struct FunctorKernel_XDeriv2D_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        // the incoming block has a padding region of 2 on each edge

        // so we have our **one** input, and one output
        auto U    = std::get<0>(views);
        auto dUdy = std::get<1>(views);

        const double h = 0.01;

        // if we're within the padding region...
        if (i > 2 && i < U.extent(0) - 2 && j > 2 && j < U.extent(1) - 2)
        {
            // then we want to run an x stencil on it, based on the left and right position
            dUdy(i, j) =
              (U(i, j - 2) - 8.0 * U(i, j - 1) + 8.0 * U(i, j + 1) - U(i, j + 2)) / (12.0 * h);
        }
#if 0
        if (j == 3)
        {
            // left edge
            dUdy(i, j) = (-3.0 * U(i, j) + 4.0 * U(i, j + 1) - U(i, j + 2)) / (2.0 * h);
        }
        else if (i == 2)
        {
            // one inside left edge
            dUdy(i, j) = (-u(i, j - 1) + u(i, j + 1)) / (2.0 * h);
        }
        else if (i == U.extent(0) - 4)
        {
            // one inside right edge
            dUdy(i, j) = (u(i, j - 1) - u(i, j + 1)) / (2.0 * h);
        }
        else if (i == U.extent(0) - 3)
        {
            // right edge
            dUdy(i, j) = (3.0 * U(i, j) - 4.0 * U(i, j - 1) + U(i, j - 2)) / (2.0 * h);
        }
#endif
    }
};


struct FunctorKernel_XDeriv2D_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        // the incoming block has a padding region of 2 on each edge

        // so we have our **one** input, and one output
        auto U    = std::get<0>(views);
        auto dUdy = std::get<1>(views);

        const double h = 0.01;

        // if we're within the padding region...
        if (i > 2 && i < U.extent(0) - 2 && j > 2 && j < U.extent(1) - 2)
        {
            // then we want to run an x stencil on it, based on the left and right position
            dUdy(i, j) =
              (U(i, j - 2) - 8.0 * U(i, j - 1) + 8.0 * U(i, j + 1) - U(i, j + 2)) / (12.0 * h);
        }
#if 0
// TODO: enable for boundary edges at some point, or add them as secondary kernels
        if (i == 3)
        {
            // left edge
            dUdy(i, j) = (-3.0 * U(i, j) + 4.0 * U(i, j + 1) - U(i, j + 2)) / (2.0 * h);
        }
        else if (i == 2)
        {
            // one inside left edge
            dUdy(i, j) = (-u(i, j - 1) + u(i, j + 1)) / (2.0 * h);
        }
        else if (i == U.extent(0) - 4)
        {
            // one inside right edge
            dUdy(i, j) = (u(i, j - 1) - u(i, j + 1)) / (2.0 * h);
        }
        else if (i == U.extent(0) - 3)
        {
            // right edge
            dUdy(i, j) = (3.0 * U(i, j) - 4.0 * U(i, j - 1) + U(i, j - 2)) / (2.0 * h);
        }
#endif
    }
};

template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelXDeriv2D(KernelOptions& options, ParameterTypes&... data_views)
{
    constexpr int KernelRank = 2;

    auto name = "y derivative, 4th order";

    auto is_const = kernel_io_map(data_views...);
    auto views_   = pack(data_views...);
    auto views    = repack_views(views_);

    auto out = std::get<1>(std::get<0>(views));

    // then set up the extent based on the output
    unsigned long N = out.extent(0);
    unsigned long M = out.extent(1);

    RangeExtent<KernelRank> extent = range_extent(Kokkos::Array<std::uint64_t, 2> { 0, 0 },
                                                  Kokkos::Array<std::uint64_t, 2> { N, M });

    auto full_policy_collection =
      make_policy_from_hyperparameters<KernelRank, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(
        extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();

    return Kernel<KernelRank,
                  FunctorKernel_XDeriv2D_Host,
                  FunctorKernel_XDeriv2D_Device,
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