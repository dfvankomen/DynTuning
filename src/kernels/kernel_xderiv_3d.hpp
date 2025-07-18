#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"

#include <Eigen/Eigen>

/**
 * First order derivative with fourth-order accuracy.
 * At boundary points it uses a lopsided second-order stencil
 *
 */
struct FunctorKernel_XDeriv3D_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views,
                                    const Index i,
                                    const Index j,
                                    const Index k) const
    {
        // the incoming block has a padding region of 2 on each edge

        // so we have our **one** input, and one output
        auto U    = std::get<0>(views);
        auto dUdx = std::get<1>(views);

        const double h = 0.01;

        // if we're within the padding region...
        if (i > 2 && i < U.extent(0) - 2 && j > 2 && j < U.extent(1) - 2 && k > 2 &&
            k < U.extent(2) - 2)
        {
            // then we want to run an x stencil on it, based on the left and right position
            dUdx(i, j, k) =
              (U(i - 2, j, k) - 8.0 * U(i - 1, j, k) + 8.0 * U(i + 1, j, k) - U(i + 2, j, k)) /
              (12.0 * h);
        }
#if 0
        if (i == 2)
        {
            // left edge
            dUdx(i, j, k) = (-3.0 * U(i, j, k) + 4.0 * U(i + 1, j, k) - U(i + 2, j, k)) / (2.0 * h);
        }
        else if (i == 3)
        {
            // one inside left edge
            dUdx(i, j, k) = (-u(i - 1, j, k) + u(i + 1)) / (2.0 * h);
        }
        else if (i == U.extent(0) - 4)
        {
            // one inside right edge
            dUdx(i, j, k) = (u(i - 1, j, k) - u(i + 1, j, k)) / (2.0 * h);
        }
        else if (i == U.extent(0) - 3)
        {
            // right edge
            dUdx(i, j, k) = (3.0 * U(i, j, k) - 4.0 * U(i - 1, j, k) + U(i - 2, j, k)) / (2.0 * h);
        }
#endif
    }
};


struct FunctorKernel_XDeriv3D_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views,
                                    const Index i,
                                    const Index j,
                                    const Index k) const
    {
        // the incoming block has a padding region of 2 on each edge

        // so we have our **one** input, and one output
        auto U    = std::get<0>(views);
        auto dUdx = std::get<1>(views);

        const double h = 0.01;

        // if we're within the padding region...
        if (i > 2 && i < U.extent(0) - 2 && j > 2 && j < U.extent(1) - 2 && k > 2 &&
            k < U.extent(2) - 2)
        {
            // then we want to run an x stencil on it, based on the left and right position
            dUdx(i, j, k) =
              (U(i - 2, j, k) - 8.0 * U(i - 1, j, k) + 8.0 * U(i + 1, j, k) - U(i + 2, j, k)) /
              (12.0 * h);
        }
#if 0
        if (i == 2)
        {
            // left edge
            dUdx(i, j, k) = (-3.0 * U(i, j, k) + 4.0 * U(i + 1, j, k) - U(i + 2, j, k)) / (2.0 * h);
        }
        else if (i == 3)
        {
            // one inside left edge
            dUdx(i, j, k) = (-u(i - 1, j, k) + u(i + 1)) / (2.0 * h);
        }
        else if (i == U.extent(0) - 4)
        {
            // one inside right edge
            dUdx(i, j, k) = (u(i - 1, j, k) - u(i + 1, j, k)) / (2.0 * h);
        }
        else if (i == U.extent(0) - 3)
        {
            // right edge
            dUdx(i, j, k) = (3.0 * U(i, j, k) - 4.0 * U(i - 1, j, k) + U(i - 2, j, k)) / (2.0 * h);
        }
#endif
    }
};

template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelXDeriv3D(KernelOptions& options, ParameterTypes&... data_views)
{
    constexpr int KernelRank = 3;

    auto name = "x derivative, 4th order";

    auto is_const = kernel_io_map(data_views...);
    auto views_   = pack(data_views...);
    auto views    = repack_views(views_);

    auto out = std::get<1>(std::get<0>(views));

    // then set up the extent based on the output
    unsigned long N = out.extent(0);
    unsigned long M = out.extent(1);
    unsigned long K = out.extent(2);

    RangeExtent<KernelRank> extent = range_extent({ 0, 0, 0 }, { N, M, K });

    auto full_policy_collection =
      make_policy_from_hyperparameters<3, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();

    std::cout << prettyprint_function_type<decltype(policy_names)>() << std::endl;
    for (std::size_t ii = 0; ii < policy_names.size(); ++ii)
    {
        std::cout << policy_names[ii] << std::endl;
    }

    return Kernel<KernelRank,
                  FunctorKernel_XDeriv3D_Host,
                  FunctorKernel_XDeriv3D_Device,
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
