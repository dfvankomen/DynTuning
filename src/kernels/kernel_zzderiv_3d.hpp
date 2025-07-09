#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"


struct FunctorKernel_ZZDeriv3D_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views,
                                    const Index i,
                                    const Index j,
                                    const Index k) const
    {
        // input (U), output (d2Udx2), spacing (for dx, dy, dz)
        auto U       = std::get<0>(views);
        auto d2Udz2  = std::get<2>(views);
        auto spacing = std::get<1>(views);

        const double dx = spacing(0);
        const double dy = spacing(1);
        const double dz = spacing(2);

        const double h        = dz;
        const double h2       = h * h;
        const double inv_12h2 = 1.0 / (12.0 * h2);

        const Index N = U.extent(0);
        const Index M = U.extent(1);
        const Index K = U.extent(2);

        // out of our "boundary" check, no need to run
        if (i >= N || j >= M || k >= K)
            return;

        // within the interior of 4th order scheme
        if (k > 1 && k < K - 2)
        {
            // then we want to run an x stencil on it, based on the left and right position
            d2Udz2(i, j, k) = (-U(i, j, k - 2) + 16.0 * U(i, j, k - 1) - 30.0 * U(i, j, k) +
                               16.0 * U(i, j, k + 1) - U(i, j, k + 2)) *
                              inv_12h2;
            return;
        }

        // left boundary
        if (k < 4)
        {
            if (k == 0)
            {
                d2Udz2(i, j, k) =
                  (2.0 * U(i, j, 0) - 5.0 * U(i, j, 1) + 4.0 * U(i, j, 2) - U(i, j, 3)) / (h2);
            }
            else if (k == 1)
            {
                d2Udz2(i, j, k) = (U(i, j, 0) - 2.0 * U(i, j, 1) + U(i, j, 2)) / (h2);
            }
            return;
        }

        // right boundary
        if (k >= K - 2)
        {
            if (k == K - 1)
            {
                d2Udz2(i, j, k) = (2.0 * U(i, j, k) - 5.0 * U(i, j, k - 1) + 4.0 * U(i, j, k - 2) -
                                   U(i, j, k - 3)) /
                                  (h2);
            }
            else if (k == K - 2)
            {
                d2Udz2(i, j, k) = (U(i, j, k - 1) - 2.0 * U(i, j, k) + U(i, j, k + 1)) / (h2);
            }
            return;
        }
    }
};

struct FunctorKernel_ZZDeriv3D_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views,
                                    const Index i,
                                    const Index j,
                                    const Index k) const
    {
        // input (U), output (d2Udx2), spacing (for dx, dy, dz)
        auto U       = std::get<0>(views);
        auto d2Udz2  = std::get<2>(views);
        auto spacing = std::get<1>(views);

        const double dx = spacing(0);
        const double dy = spacing(1);
        const double dz = spacing(2);

        const double h        = dz;
        const double h2       = h * h;
        const double inv_12h2 = 1.0 / (12.0 * h2);

        const Index N = U.extent(0);
        const Index M = U.extent(1);
        const Index K = U.extent(2);

        // out of our "boundary" check, no need to run
        if (i >= N || j >= M || k >= K)
            return;

        // within the interior of 4th order scheme
        if (k > 1 && k < K - 2)
        {
            // then we want to run an x stencil on it, based on the left and right position
            d2Udz2(i, j, k) = (-U(i, j, k - 2) + 16.0 * U(i, j, k - 1) - 30.0 * U(i, j, k) +
                               16.0 * U(i, j, k + 1) - U(i, j, k + 2)) *
                              inv_12h2;
            return;
        }

        // left boundary
        if (k < 4)
        {
            if (k == 0)
            {
                d2Udz2(i, j, k) =
                  (2.0 * U(i, j, 0) - 5.0 * U(i, j, 1) + 4.0 * U(i, j, 2) - U(i, j, 3)) / (h2);
            }
            else if (k == 1)
            {
                d2Udz2(i, j, k) = (U(i, j, 0) - 2.0 * U(i, j, 1) + U(i, j, 2)) / (h2);
            }
            return;
        }

        // right boundary
        if (k >= K - 2)
        {
            if (k == K - 1)
            {
                d2Udz2(i, j, k) = (2.0 * U(i, j, k) - 5.0 * U(i, j, k - 1) + 4.0 * U(i, j, k - 2) -
                                   U(i, j, k - 3)) /
                                  (h2);
            }
            else if (k == K - 2)
            {
                d2Udz2(i, j, k) = (U(i, j, k - 1) - 2.0 * U(i, j, k) + U(i, j, k + 1)) / (h2);
            }
            return;
        }
    }
};

template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelZZDeriv3D(KernelOptions& options, ParameterTypes&... data_views)
{
    constexpr int KernelRank = 3;

    auto name = "zz derivative, 3d, 4th order";

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
                  FunctorKernel_ZZDeriv3D_Host,
                  FunctorKernel_ZZDeriv3D_Device,
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
