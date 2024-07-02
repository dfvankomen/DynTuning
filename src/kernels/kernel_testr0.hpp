#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"

struct FunctorKernelRank0_Host
{
    template<typename ViewsTuple>
    KOKKOS_FUNCTION void operator()(ViewsTuple views) const
    {
        auto a = std::get<0>(views);
        auto b = std::get<1>(views);
        auto c = std::get<2>(views);

        for (size_t j = 0; j < a.extent(0); j++)
        {
            c(j) = a(j) * b(j);
        }
    }
};


struct FunctorKernelRank0_Device
{
    template<typename ViewsTuple>
    KOKKOS_FUNCTION void operator()(ViewsTuple views) const
    {

        auto a = std::get<0>(views);
        auto b = std::get<1>(views);
        auto c = std::get<2>(views);

        auto name               = "vector-vector multiply (alt)";
        using DeviceRangePolicy = typename RangePolicy<1, DeviceExecutionSpace>::type;
        auto range_policy       = DeviceRangePolicy(0, c.extent(0));

        Kokkos::parallel_for(name, range_policy, KOKKOS_LAMBDA(int j) { c(j) = a(j) * b(j); });
    }
};

#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))

template<typename... ParameterTypes>
inline auto KernelTestR0(KernelOptions& options, ParameterTypes&... data_views)
{
    auto name = "vector-vector multiply (alt)";

    auto is_const = kernel_io_map(data_views...);

    auto views_ = pack(data_views...);

    auto views = repack_views(views_);

    auto out    = std::get<2>(std::get<0>(views));
    auto extent = range_extent();

    return Kernel<0,
                  FunctorKernelRank0_Host,
                  FunctorKernelRank0_Device,
                  decltype(views),
                  decltype(is_const)>(name, views, is_const, extent, options);
}
