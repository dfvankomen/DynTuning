#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "range.hpp"
#include "view.hpp"

#include <cstddef>
#include <utility>

struct KernelOptions
{
    std::vector<DeviceSelector> devices;
};

//=============================================================================
// Kernel
//=============================================================================

// template<int KernelRank, class FunctorType, typename... ParameterTypes>
template<int KernelRank,
         class HostFunctorType,
         class DeviceFunctorType,
         typename DataViewsType,
         typename IsConstTupleType>
//  typename HostExecutionPolicyCollection,
//  typename DeviceExecutionPolicyCollection>
class Kernel
{
  public:
    static constexpr int rank = KernelRank;

    // Note: we are choosing the host and device excution space at compile time
    using HostExecutionSpace   = Kokkos::KOKKOS_HOST;
    using DeviceExecutionSpace = Kokkos::KOKKOS_DEVICE;
    using BoundType            = RangeExtent<KernelRank>::value_type;
    using HostRangePolicy      = typename RangePolicy<KernelRank, HostExecutionSpace>::type;
    using DeviceRangePolicy    = typename RangePolicy<KernelRank, DeviceExecutionSpace>::type;

    Kernel(const char* name,
           DataViewsType views,
           IsConstTupleType is_const,
           const RangeExtent<KernelRank>& extent,
           KernelOptions& options
           //    HostExecutionPolicyCollection& host_execution_policies,
           //    DeviceExecutionPolicyCollection& device_execution_policies)
           )
      : kernel_name_(std::string(name))
      , data_views_(views)
      , is_const_(is_const)
      , range_lower_(extent.lower)
      , range_upper_(extent.upper)
      , range_policy_host_(HostRangePolicy(extent.lower, extent.upper))
      , range_policy_device_(DeviceRangePolicy(extent.lower, extent.upper))
      , options_(options)
    //   , host_execution_policies_(host_execution_policies)
    //   , device_execution_policies_(device_execution_policies)
    {
    }

    void operator()(DeviceSelector device_selector)
    {
        call_kernel(*this, device_selector);
    };

    // kernel name for debugging
    std::string kernel_name_;

    // The kernel code that will be called in an executation on the respective views
    HostFunctorType kernel_functor_host_;
    DeviceFunctorType kernel_functor_device_;

    DataViewsType data_views_;
    IsConstTupleType is_const_;

    // Properties pertaining to range policy
    const BoundType range_lower_;
    const BoundType range_upper_;
    // tile_type tile_;
    HostRangePolicy range_policy_host_;
    DeviceRangePolicy range_policy_device_;

    // Execution Options
    KernelOptions& options_;

    // the ranges
    // HostExecutionPolicyCollection host_execution_policies_;
    // DeviceExecutionPolicyCollection device_execution_policies_;
};


//=============================================================================
// Helpers
//=============================================================================

template<int KernelRank, typename ViewsType, typename RangePolicyType, typename FunctorType>
inline void call_kernel(const std::string& name,
                        const RangePolicyType& range_policy,
                        ViewsType& views,
                        const FunctorType functor)
{
    if constexpr (KernelRank == 0)
    {
        // just call the functor, they only get views
        functor(views);
    }
    else if constexpr (KernelRank == 1)
    {
        Kokkos::parallel_for(name, range_policy, KOKKOS_LAMBDA(int i) { functor(views, i); });
    }
    else if constexpr (KernelRank == 2)
    {
        Kokkos::parallel_for(name, range_policy, KOKKOS_LAMBDA(int i, int j) {
            functor(views, i, j);
        });
    }
}

template<typename KernelType>
inline void call_kernel(KernelType& k, DeviceSelector device_selector)
{
    if (device_selector == DeviceSelector::HOST)
    {
        call_kernel<KernelType::rank>(k.kernel_name_,
                                      k.range_policy_host_,
                                      std::get<0>(k.data_views_),
                                      k.kernel_functor_host_);
    }
    else if (device_selector == DeviceSelector::DEVICE)
    {
        call_kernel<KernelType::rank>(k.kernel_name_,
                                      k.range_policy_device_,
                                      std::get<1>(k.data_views_),
                                      k.kernel_functor_device_);
    }
}

template<typename... T>
inline auto kernel_io_map(T&... args)
{
    return std::make_tuple(
      static_cast<bool>(std::is_const_v<std::remove_reference_t<decltype(args)>>)...);
}



struct MyTestObject
{
    std::size_t maxthreads;
    std::size_t minblocks;
};


constexpr auto most_internal_linspace_hi(double start,
                                         double end,
                                         std::size_t nsteps,
                                         std::size_t i)
{
    return static_cast<std::size_t>(start + i * ((end - start) / (nsteps - 1)));
}

constexpr auto linspace_inner_inner(double start_thread,
                                    double end_thread,
                                    double start_block,
                                    double end_block,
                                    std::size_t n_thread,
                                    std::size_t n_block,
                                    std::size_t k)
{
    std::size_t i = k / n_block;
    std::size_t j = k % n_block;

    // then we can create our object
    return MyTestObject({ most_internal_linspace_hi(start_thread, end_thread, n_thread, i),
                          most_internal_linspace_hi(start_block, end_block, n_block, j) });
}

template<std::size_t... I>
constexpr auto linspace_unrolled_inner(double start_thread,
                             double end_thread,
                             double start_block,
                             double end_block,
                             std::size_t n_thread,
                             std::size_t n_block,
                             std::index_sequence<I...>)
{
    return std::make_tuple(linspace_inner_inner(start_thread,
                                                end_thread,
                                                start_block,
                                                end_block,
                                                n_thread,
                                                n_block,
                                                I)...);
}

template<std::size_t N, std::size_t M>
constexpr auto linspace_unrolled(double start_thread,
                                 double end_thread,
                                 double start_block,
                                 double end_block)
{
    static_assert(N > 1,
                  "Number of steps has to be greater than 1 for linspace calculations to work!");
    static_assert(M > 1,
                  "Number of steps has to be greater than 1 for linspace calculations to work!");

    // now we can call the first layer
    return linspace_unrolled_inner(start_thread,
                                   end_thread,
                                   start_block,
                                   end_block,
                                   N,
                                   M,
                                   std::make_index_sequence<N * M> {});
}
