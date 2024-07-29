
#pragma once

#include "Eigen"
#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"


struct FunctorKernel_MatSum_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        // so we have our **two** inputs, and one output
        auto A = std::get<0>(views);
        auto B = std::get<1>(views);
        auto C = std::get<2>(views);

        // just add them together!
        C(i, j) = A(i, j) + B(i, j);
    }
};


struct FunctorKernel_MatSum_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        // so we have our **two** inputs, and one output
        auto A = std::get<0>(views);
        auto B = std::get<1>(views);
        auto C = std::get<2>(views);

        // just add them together!
        C(i, j) = A(i, j) + B(i, j);
    }
};

template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelMatSum(KernelOptions& options, ParameterTypes&... data_views)
{
    constexpr int KernelRank = 2;

    auto name = "grad (two matrix sum)";

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
                  FunctorKernel_MatSum_Host,
                  FunctorKernel_MatSum_Device,
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
