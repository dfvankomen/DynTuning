#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "range.hpp"
#include "view.hpp"

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
           KernelOptions& options)
      : kernel_name_(std::string(name))
      , data_views_(views)
      , is_const_(is_const)
      , range_lower_(extent.lower)
      , range_upper_(extent.upper)
      , range_policy_host_(HostRangePolicy(extent.lower, extent.upper))
      , range_policy_device_(DeviceRangePolicy(extent.lower, extent.upper))
      , options_(options)
    {
#ifdef NDEBUG
        // debugging diagnostics
        printf("\nHost Execution Space:\n");
        HostExecutionSpace {}.print_configuration(std::cout);
        printf("\nDevice Execution Space:\n");
        DeviceExecutionSpace {}.print_configuration(std::cout);
#endif
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
    if constexpr (KernelRank == 1)
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