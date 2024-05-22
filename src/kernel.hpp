#pragma once

#include "common.hpp"
#include "range.hpp"
#include "view.hpp"
#include "Kokkos_Core.hpp"

struct KernelOptions
{
    std::vector<DeviceSelector> devices;
};

//=============================================================================
// Kernel
//=============================================================================

//template<int KernelRank, class FunctorType, typename... ParameterTypes>
template<int KernelRank, class FunctorType, typename DataViewsType, typename IsConstTupleType>
class Kernel
{
  public:
    static constexpr int rank = KernelRank;

    // Note: we are choosing the host and device excution space at compile time
    using HostExecutionSpace   = Kokkos::KOKKOS_HOST;
    using DeviceExecutionSpace = Kokkos::KOKKOS_DEVICE;
    using BoundType            = RangeExtent<KernelRank>::value_type;
    //using DataTuple            = std::tuple<ParameterTypes&...>;
    using HostRangePolicy      = typename RangePolicy<KernelRank, HostExecutionSpace>::type;
    using DeviceRangePolicy    = typename RangePolicy<KernelRank, DeviceExecutionSpace>::type;
    //using DataParamsType       = std::tuple<ParameterTypes&...>;
    
    //using DeviceDataViewsType =
    //  std::tuple<typename EquivalentView<DeviceExecutionSpace, ParameterTypes>::type...>;
    //using HostDataViewsType = std::tuple<
    //  typename EquivalentView<DeviceExecutionSpace, ParameterTypes>::type::HostMirror...>;
    
    // START HERE, how do we set this type at compile time when both the layout and param type will vary?
    //using HostDataViewsType = 
    //    std::tuple<typename EquivalentView<HostExecutionSpace, HostExecutionSpace::array_layout, ParameterTypes>::type...>;
    //using tmpDataViewsType = 
    //    std::tuple<typename EquivalentView<HostExecutionSpace, Kokkos::LayoutLeft, ParameterTypes>::type...>;
    //using DeviceDataViewsType =
    //    std::tuple<typename EquivalentView<DeviceExecutionSpace, Kokkos::LayoutLeft, ParameterTypes>::type...>;
    //
    //using tmptype = Views<HostExecutionSpace, ViewMemoryType::NONOWNING>;

    //using DataViewsType = std::tuple<ParameterTypes&...>;
    
    Kernel(const char* name,
           DataViewsType views,
           IsConstTupleType is_const,
           const RangeExtent<KernelRank>& extent,
           KernelOptions& options)
      : kernel_name_(std::string(name))
      , data_views_(views)
      , is_const_(is_const)
      //, data_params_(params)
      //, data_views_device_(Views<DeviceExecutionSpace>::create_views_from_tuple(params))
      //, data_views_host_(
      //    Views<DeviceExecutionSpace>::create_mirror_views_from_tuple(data_views_device_))
      //, data_views_host_(Views<HostExecutionSpace, ViewMemoryType::NONOWNING>::create_views_from_tuple(params)) // non-owning
      //, data_views_host_(tmptype::create_views_from_tuple(params)) // non-owning
      //, data_views_tmp_(Views<HostExecutionSpace, ViewMemoryType::TMP>::create_views_from_tuple(params)) // owning temp space
      //, data_views_device_(Views<DeviceExecutionSpace, ViewMemoryType::OWNING>::create_views_from_tuple(params)) // owning
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
    FunctorType kernel_functor_;

    DataViewsType data_views_;
    IsConstTupleType is_const_;

    // Data parameters and views thereof (in the execution spaces that will be considered)
    //DataParamsType data_params_;
    
    // Note: this has to go first for initialization to function correctly.
    //HostDataViewsType data_views_host_;
    //tmpDataViewsType data_views_tmp_;
    //DeviceDataViewsType data_views_device_;    

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
        Kokkos::parallel_for(
            name,
            range_policy,
            KOKKOS_LAMBDA(int i) { functor(views, i); }
        );
    }
    else if constexpr (KernelRank == 2)
    {
        Kokkos::parallel_for(
            name,
            range_policy,
            KOKKOS_LAMBDA(int i, int j) { functor(views, i, j); });
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
                                      k.kernel_functor_);
    }
    else if (device_selector == DeviceSelector::DEVICE)
    {
        call_kernel<KernelType::rank>(k.kernel_name_,
                                      k.range_policy_device_,
                                      std::get<1>(k.data_views_),
                                      k.kernel_functor_);
    }
}