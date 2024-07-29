#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"

#include <Kokkos_Macros.hpp>

struct FunctorBranchTest2d_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        auto a = std::get<0>(views);
        auto b = std::get<1>(views);

        // branch based on even/odd
        if (i % 2 == 0)
        {
            if (j % 2 == 0)
                b(i, j) = a(i, j);
            else
                b(i, j) = a(i, j) * b(i, j);
        }
        else
        {
            if (j % 2 == 0)
                b(i, j) = a(i, j);
            else
                b(i, j) = a(i, j) * b(i, j);
        }
    }
};

struct FunctorBranchTest2d_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        auto a = std::get<0>(views);
        auto b = std::get<1>(views);

        // branch based on even/odd
        if (i % 2 == 0)
        {
            if (j % 2 == 0)
                b(i, j) = a(i, j);
            else
                b(i, j) = a(i, j) * b(i, j);
        }
        else
        {
            if (j % 2 == 0)
                b(i, j) = a(i, j);
            else
                b(i, j) = a(i, j) * b(i, j);
        }
    }
};


template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelBranchTest2D(KernelOptions& options, ParameterTypes&... data_views)
{
    constexpr int KernelRank = 2;
    auto name                = "kernel branch test 1d";

    auto is_const = kernel_io_map(data_views...);

    // WARNING: Again, remapping loses const!
    auto views_ = pack(data_views...);

    auto views = repack_views(views_);

    // then set the extent
    auto out = std::get<1>(std::get<0>(views));
    RangeExtent<KernelRank> extent =
      range_extent(Kokkos::Array<std::uint64_t, 2> { 0, 0 },
                   Kokkos::Array<std::uint64_t, 2> { out.extent(0), out.extent(1) });

    auto full_policy_collection =
      make_policy_from_hyperparameters<KernelRank, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(
        extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();

    return Kernel<2,
                  FunctorBranchTest2d_Host,
                  FunctorBranchTest2d_Device,
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