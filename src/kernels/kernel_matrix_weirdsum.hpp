
#pragma once

#include "Eigen"
#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"

#include <cstddef>

struct FunctorKernel_WeirdMatSum_Host
{
    // so, this one is the host and it doesn't do CUBLAS, it just uses regular blas I think
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        auto A = std::get<0>(views);
        auto B = std::get<1>(views);

        double sum = 0.0;

        for (std::size_t ii = 0; ii < A.extent(0); ++ii)
        {
            for (std::size_t jj = 0; jj < A.extent(1); ++jj)
            {
                sum += A(i, j);
            }
        }
        B(i, j) = sum;
    }
};

struct FunctorKernel_WeirdMatSum_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        auto A = std::get<0>(views);
        auto B = std::get<1>(views);

        double sum = 0.0;

        for (std::size_t ii = 0; ii < A.extent(0); ++ii)
        {
            for (std::size_t jj = 0; jj < A.extent(1); ++jj)
            {
                sum += A(i, j);
            }
        }
        B(i, j) = sum;
    }
};


template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelMatWeirdSum(KernelOptions& options, ParameterTypes&... data_views)
{
    constexpr int KernelRank = 2;

    auto name = "weird matrix sum";

    auto is_const = kernel_io_map(data_views...);

    // WARNING, remapping loses const!
    auto views_ = pack(data_views...);

    auto views = repack_views(views_);
    // first dim you pull out is host/scratch/device, second is variable

    // get the extent based on the host view of the matrix
    auto out = std::get<2>(std::get<0>(views));

    // we'll assume that the user has properly set up the right matrix sizes, so the iterations are
    // only in the out to store
    unsigned long N = out.extent(0);
    unsigned long M = out.extent(1);

    RangeExtent<KernelRank> extent = range_extent(Kokkos::Array<std::uint64_t, 2> { 0, 0 },
                                                  Kokkos::Array<std::uint64_t, 2> { N, M });

    // generate the policy collection based on user hyperparameters
    auto full_policy_collection =
      make_policy_from_hyperparameters<KernelRank, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(
        extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();

    return Kernel<KernelRank,
                  FunctorKernel_WeirdMatSum_Host,
                  FunctorKernel_WeirdMatSum_Device,
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