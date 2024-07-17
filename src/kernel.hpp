#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "range.hpp"
#include "view.hpp"

#include <cstddef>
#include <ostream>
#include <utility>

struct KernelOptions
{
    std::vector<DeviceSelector> devices;
};


struct HyperParameterStorage
{
    std::size_t threads;
    std::size_t blocks;

    inline std::string to_string() const
    {
        return "t,b: " + std::to_string(threads) + "," + std::to_string(blocks);
    }
    inline std::string to_string_csv() const
    {
        return std::to_string(threads) + "," + std::to_string(blocks);
    }
};

inline std::ostream& operator<<(std::ostream& os, const HyperParameterStorage& hp)
{
    os << "threads: " << hp.threads << " blocks: " << hp.blocks;
    return os;
}


template<unsigned int StartThreads,
         unsigned int EndThreads,
         unsigned int NThreads,
         unsigned int StartBlocks,
         unsigned int EndBlocks,
         unsigned int NBlocks>
struct LinspaceOptions
{
    static constexpr unsigned int sthreads = StartThreads;
    static constexpr unsigned int ethreads = EndThreads;
    static constexpr unsigned int nthreads = NThreads;
    static constexpr unsigned int sblocks  = StartBlocks;
    static constexpr unsigned int eblocks  = EndBlocks;
    static constexpr unsigned int nblocks  = NBlocks;
};

//=============================================================================
// Kernel
//=============================================================================

// template<int KernelRank, class FunctorType, typename... ParameterTypes>
template<int KernelRank,
         class HostFunctorType,
         class DeviceFunctorType,
         typename DataViewsType,
         typename IsConstTupleType,
         typename DeviceExecutionPolicyCollection,
         typename DeviceExecutionPolicyData>
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
    }

    Kernel(const char* name,
           DataViewsType views,
           IsConstTupleType is_const,
           const RangeExtent<KernelRank>& extent,
           KernelOptions& options,
           DeviceExecutionPolicyCollection& device_execution_policies,
           DeviceExecutionPolicyData& policy_data)
      : kernel_name_(std::string(name))
      , data_views_(views)
      , is_const_(is_const)
      , range_lower_(extent.lower)
      , range_upper_(extent.upper)
      , range_policy_host_(HostRangePolicy(extent.lower, extent.upper))
      , range_policy_device_(DeviceRangePolicy(extent.lower, extent.upper))
      , options_(options)
      , device_execution_policies_(device_execution_policies)
      , device_execution_policy_data_(policy_data)
    {
    }

    void operator()(DeviceSelector device_selector, std::size_t rpIdx = 0)
    {
        call_kernel(*this, device_selector, rpIdx);
    };

    auto get_hyperparameters(DeviceSelector device_selector, std::size_t rpIdx = 0)
    {

        if (device_selector == DeviceSelector::DEVICE)
        {
            return device_execution_policy_data_[rpIdx];
        }
        else
        {
            return HyperParameterStorage({ 0, 0 });
        }
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
    DeviceExecutionPolicyCollection device_execution_policies_;
    std::size_t n_device_execution_policies_ = std::tuple_size_v<DeviceExecutionPolicyCollection>;

    DeviceExecutionPolicyData device_execution_policy_data_;
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
inline void call_kernel(KernelType& k, DeviceSelector device_selector, std::size_t idx = 0)
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
        if constexpr (std::tuple_size_v<decltype(k.device_execution_policies_)> > 1)
        {
            find_tuple(k.device_execution_policies_,
                       idx,
                       [&]<typename KernelExecutionPolicyType>(
                         KernelExecutionPolicyType& range_policy_device)
            {
                call_kernel<KernelType::rank>(k.kernel_name_,
                                              range_policy_device,
                                              std::get<1>(k.data_views_),
                                              k.kernel_functor_device_);
            });
        }
        else
        {
            call_kernel<KernelType::rank>(k.kernel_name_,
                                          std::get<0>(k.device_execution_policies_),
                                          std::get<1>(k.data_views_),
                                          k.kernel_functor_device_);
        }
    }
}

template<typename... T>
inline auto kernel_io_map(T&... args)
{
    return std::make_tuple(
      static_cast<bool>(std::is_const_v<std::remove_reference_t<decltype(args)>>)...);
}



constexpr auto linspace_val(double start, double end, unsigned int nsteps, unsigned int i)
{
    return static_cast<unsigned int>(start + i * ((end - start) / (nsteps - 1)));
}


template<int KernelRank, typename ExecutionSpace>
constexpr auto _crpd_innermost_default(const RangeExtent<KernelRank>& extent)
{
    using DeviceRangePolicy = typename RangePolicy<KernelRank, ExecutionSpace>::type;

    // then we can create our object
    return DeviceRangePolicy(extent.lower, extent.upper);
}

template<int KernelRank, typename ExecutionSpace, typename LinspaceOptions, unsigned int K>
constexpr auto _crpd_innermost(const RangeExtent<KernelRank>& extent)
{
    constexpr unsigned int threads = linspace_val(LinspaceOptions::sthreads,
                                                  LinspaceOptions::ethreads,
                                                  LinspaceOptions::nthreads,
                                                  K / LinspaceOptions::nblocks);
    constexpr unsigned int blocks  = linspace_val(LinspaceOptions::sblocks,
                                                 LinspaceOptions::eblocks,
                                                 LinspaceOptions::nblocks,
                                                 K % LinspaceOptions::nblocks);
    using ComputedLaunchBounds     = typename Kokkos::LaunchBounds<threads, blocks>;
    using DeviceRangePolicy =
      typename RangePolicy<KernelRank, ExecutionSpace, ComputedLaunchBounds>::type;

    // then we can create our object
    // return MyTestObject({ threads, blocks });
    return DeviceRangePolicy(extent.lower, extent.upper);
}

template<int KernelRank, typename ExecutionSpace, typename LinspaceOptions, std::size_t... I>
constexpr auto _crpd_inner(const RangeExtent<KernelRank>& extent, std::index_sequence<I...>)
{
    // create a tuple that holds everything we need
    // the first one should include Kokkos' automatic just in case
    return std::make_tuple(
      _crpd_innermost_default<KernelRank, ExecutionSpace>(extent),
      _crpd_innermost<KernelRank, ExecutionSpace, LinspaceOptions, I>(extent)...);
}

template<int KernelRank, typename ExecutionSpace, typename LinspaceOptions>
constexpr auto create_range_policy_device(const RangeExtent<KernelRank>& extent)
{
    static_assert(LinspaceOptions::nthreads > 1,
                  "Number of steps has to be greater than 1 for linspace calculations to work!");
    static_assert(LinspaceOptions::nblocks > 1,
                  "Number of steps has to be greater than 1 for linspace calculations to work!");

    // now we can call the first layer
    return _crpd_inner<KernelRank, ExecutionSpace, LinspaceOptions>(
      extent,
      std::make_index_sequence<LinspaceOptions::nthreads * LinspaceOptions::nblocks> {});
}


template<int KernelRank, typename ExecutionSpace = Kokkos::KOKKOS_DEVICE>
constexpr auto create_range_policy_device(const RangeExtent<KernelRank>& extent)
{
    using DeviceRangePolicy = typename RangePolicy<KernelRank, ExecutionSpace>::type;

    // then we can create our object
    // return MyTestObject({ threads, blocks });
    return std::make_tuple(DeviceRangePolicy(extent.lower, extent.upper));
}


// Linspace Hyper Parameter Storage
template<typename LinspaceOptions, unsigned int K>
constexpr auto _crpdc_innermost()
{
    constexpr unsigned int threads = linspace_val(LinspaceOptions::sthreads,
                                                  LinspaceOptions::ethreads,
                                                  LinspaceOptions::nthreads,
                                                  K / LinspaceOptions::nblocks);
    constexpr unsigned int blocks  = linspace_val(LinspaceOptions::sblocks,
                                                 LinspaceOptions::eblocks,
                                                 LinspaceOptions::nblocks,
                                                 K % LinspaceOptions::nblocks);

    return HyperParameterStorage({ threads, blocks });
}

template<typename LinspaceOptions, std::size_t... I>
constexpr auto _crpdc_inner(std::index_sequence<I...>)
{
    // create a tuple that holds everything we need
    // the first one should include Kokkos' automatic just in case
    std::vector<HyperParameterStorage> output;

    // start by pushing back the rest of it

    output.push_back(HyperParameterStorage({ 0, 0 }));
    (output.push_back(_crpdc_innermost<LinspaceOptions, I>()), ...);

    return output;
    // return std::make_tuple(HyperParameterStorage({ 0, 0 }),
    //                        _crpdc_innermost<LinspaceOptions, I>()...);
}

template<typename LinspaceOptions>
constexpr auto create_range_policy_device_collection()
{
    static_assert(LinspaceOptions::nthreads > 1,
                  "Number of steps has to be greater than 1 for linspace calculations to work!");
    static_assert(LinspaceOptions::nblocks > 1,
                  "Number of steps has to be greater than 1 for linspace calculations to work!");

    // now we can call the first layer
    return _crpdc_inner<LinspaceOptions>(
      std::make_index_sequence<LinspaceOptions::nthreads * LinspaceOptions::nblocks> {});
}


inline auto create_range_policy_device_collection()
{
    // just return a vector with one type
    return std::vector<HyperParameterStorage>({ HyperParameterStorage({ 0, 0 }) });
}