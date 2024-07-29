#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"

#include <Kokkos_Macros.hpp>

struct FunctorBranchTest_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i) const
    {
        auto a = std::get<0>(views);
        auto b = std::get<1>(views);
        b(i)   = 1;

        // // branch based on even/odd
        // if (i % 2 == 0)
        //     b(i) = a(i);
        // else
        //     b(i) = a(i) * b(i);
    }
};

struct FunctorBranchTest_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i) const
    {
        auto a = std::get<0>(views);
        auto b = std::get<1>(views);
        b(i)   = 1;

        // // branch based on even/odd
        // if (i % 2 == 0)
        //     b(i) = a(i);
        // else
        //     b(i) = a(i) * b(i);
    }
};


template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelBranchTest(KernelOptions& options, ParameterTypes&... data_views)
{
    auto name = "branch test 1d";

    auto is_const = kernel_io_map(data_views...);

    // WARNING, remapping loses const! Fix this later...
    // transpose the views tuple
    // views_: i = variable, j = device
    // views:  i = device, j = variable
    auto views_ = pack(data_views...);

    auto views = repack_views(views_);

    // set the extent based on the host view of the output
    auto out    = std::get<1>(std::get<0>(views));
    auto extent = range_extent(0, out.extent(0));

    // generate the policy collection based on user hyperparameters
    auto full_policy_collection =
      make_policy_from_hyperparameters<1, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();


    // create the kernel
    return Kernel<1,
                  FunctorBranchTest_Host,
                  FunctorBranchTest_Device,
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
