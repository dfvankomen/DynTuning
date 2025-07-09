#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"


struct FunctorKernel_YYDeriv3D_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views,
                                    const Index i,
                                    const Index j,
                                    const Index k) const
    {
        // input (U), output (d2Udy2), spacing (for dx, dy, dz)
        auto U       = std::get<0>(views);
        auto d2Udy2  = std::get<2>(views);
        auto spacing = std::get<1>(views);

        const double dx = spacing(0);
        const double dy = spacing(1);
        const double dz = spacing(2);

        const double h        = dy;
        const double h2       = h * h;
        const double inv_12h2 = 1.0 / (12.0 * h2);

        const Index N = U.extent(0);
        const Index M = U.extent(1);
        const Index K = U.extent(2);

        // out of our "boundary" check, no need to run
        if (i >= N || j >= M || k >= K)
            return;

        // within the interior of 4th order scheme
        if (j > 1 && j < M - 2)
        {
            // then we want to run an x stencil on it, based on the left and right position
            d2Udy2(i, j, k) = (-U(i, j - 2, k) + 16.0 * U(i, j - 1, k) - 30.0 * U(i, j, k) +
                               16.0 * U(i, j + 1, k) - U(i, j + 2, k)) *
                              inv_12h2;
            return;
        }

        // left boundary
        if (j < 4)
        {
            if (j == 0)
            {
                d2Udy2(i, j, k) =
                  (2.0 * U(i, 0, k) - 5.0 * U(i, 1, k) + 4.0 * U(i, 2, k) - U(i, 3, k)) / (h2);
            }
            else if (j == 1)
            {
                d2Udy2(i, j, k) = (U(i, 0, k) - 2.0 * U(i, 1, k) + U(i, 2, k)) / (h2);
            }
            return;
        }

        // right boundary
        if (j >= M - 2)
        {
            if (j == M - 1)
            {
                d2Udy2(i, j, k) = (2.0 * U(i, j, k) - 5.0 * U(i, j - 1, k) + 4.0 * U(i, j - 2, k) -
                                   U(i, j - 3, k)) /
                                  (h2);
            }
            else if (j == M - 2)
            {
                d2Udy2(i, j, k) = (U(i, j - 1, k) - 2.0 * U(i, j, k) + U(i, j + 1, k)) / (h2);
            }
            return;
        }
    }
};

struct FunctorKernel_YYDeriv3D_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views,
                                    const Index i,
                                    const Index j,
                                    const Index k) const
    {
        // input (U), output (d2Udy2), spacing (for dx, dy, dz)
        auto U       = std::get<0>(views);
        auto d2Udy2  = std::get<2>(views);
        auto spacing = std::get<1>(views);

        const double dx = spacing(0);
        const double dy = spacing(1);
        const double dz = spacing(2);

        const double h        = dy;
        const double h2       = h * h;
        const double inv_12h2 = 1.0 / (12.0 * h2);

        const Index N = U.extent(0);
        const Index M = U.extent(1);
        const Index K = U.extent(2);

        // out of our "boundary" check, no need to run
        if (i >= N || j >= M || k >= K)
            return;

        // within the interior of 4th order scheme
        if (j > 1 && j < M - 2)
        {
            // then we want to run an x stencil on it, based on the left and right position
            d2Udy2(i, j, k) = (-U(i, j - 2, k) + 16.0 * U(i, j - 1, k) - 30.0 * U(i, j, k) +
                               16.0 * U(i, j + 1, k) - U(i, j + 2, k)) *
                              inv_12h2;
            return;
        }

        // left boundary
        if (j < 4)
        {
            if (j == 0)
            {
                d2Udy2(i, j, k) =
                  (2.0 * U(i, 0, k) - 5.0 * U(i, 1, k) + 4.0 * U(i, 2, k) - U(i, 3, k)) / (h2);
            }
            else if (j == 1)
            {
                d2Udy2(i, j, k) = (U(i, 0, k) - 2.0 * U(i, 1, k) + U(i, 2, k)) / (h2);
            }
            return;
        }

        // right boundary
        if (j >= M - 2)
        {
            if (j == M - 1)
            {
                d2Udy2(i, j, k) = (2.0 * U(i, j, k) - 5.0 * U(i, j - 1, k) + 4.0 * U(i, j - 2, k) -
                                   U(i, j - 3, k)) /
                                  (h2);
            }
            else if (j == M - 2)
            {
                d2Udy2(i, j, k) = (U(i, j - 1, k) - 2.0 * U(i, j, k) + U(i, j + 1, k)) / (h2);
            }
            return;
        }
    }
};

template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelYYDeriv3D(KernelOptions& options, ParameterTypes&... data_views)
{
    constexpr int KernelRank = 3;

    auto name = "yy derivative, 3d, 4th order";

    auto is_const = kernel_io_map(data_views...);
    auto views_   = pack(data_views...);
    auto views    = repack_views(views_);

    auto out = std::get<2>(std::get<0>(views));

    // then set up the extent based on the output
    unsigned long N = out.extent(0);
    unsigned long M = out.extent(1);
    unsigned long K = out.extent(2);

    RangeExtent<KernelRank> extent = range_extent(Kokkos::Array<std::uint64_t, 3> { 0, 0, 0 },
                                                  Kokkos::Array<std::uint64_t, 3> { N, M, K });

    auto full_policy_collection =
      make_policy_from_hyperparameters<KernelRank, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(
        extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();

    return Kernel<KernelRank,
                  FunctorKernel_YYDeriv3D_Host,
                  FunctorKernel_YYDeriv3D_Device,
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
