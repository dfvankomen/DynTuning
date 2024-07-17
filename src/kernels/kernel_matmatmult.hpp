#pragma once

#include "Eigen"
#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"

// just the traditional implementation, NOT FAST

struct FunctorMatMatMult_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        // certainly want some eigen implementation as well
        auto A = std::get<0>(views);
        auto B = std::get<1>(views);
        auto C = std::get<2>(views);

        double sum = 0.0;

        for (size_t k = 0; k < A.extent(1); k++)
        {
            sum += A(i, k) * B(k, j);
        }

        C(i, j) = sum;
    }
};

struct FunctorMatMatMult_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        // maybe just straight up CuBLAS?
        auto A = std::get<0>(views);
        auto B = std::get<1>(views);
        auto C = std::get<2>(views);

        double sum = 0.0;

        for (size_t k = 0; k < A.extent(1); k++)
        {
            sum += A(i, k) * B(k, j);
        }

        C(i, j) = sum;
    }
};

#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))

template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelMatMatMult(KernelOptions& options, ParameterTypes&... data_views)
{
    auto name = "matrix-matrix multiply";

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

    auto extent = range_extent({ 0, 0 }, { N, M });

    // generate the policy collection based on user hyperparameters
    auto full_policy_collection =
      make_policy_from_hyperparameters<2, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();

    return Kernel<2,
                  FunctorMatMatMult_Host,
                  FunctorMatMatMult_Device,
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