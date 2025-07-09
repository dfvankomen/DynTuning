#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "range.hpp"
#include "view.hpp"

#include <cstddef>
#include <ostream>
#include <type_traits>
#include <utility>

/**
 * @brief Struct to store kernel options
 *
 */
struct KernelOptions
{
    // List of devices to use
    std::vector<DeviceSelector> devices;
};


/**
 * @brief Hyperparameter Storage
 *
 * This is used to store what hyperparameters a given kernel needs.
 * It's hard to get the parameters at runtime, so this struct is made to store
 * what the compiler decided on so it can be easily accessed without templating.
 */
struct HyperParameterStorage
{
    // Number of threads to use
    std::size_t threads;
    // Number of blocks to use
    std::size_t blocks;

    /**
     * @brief Convert struct to string
     *
     * @return std::string Output string
     */
    inline std::string to_string() const
    {
        return "t,b: " + std::to_string(threads) + "," + std::to_string(blocks);
    }
    /**
     * @brief Helper function to convert struct to string for CSV
     *
     * @return std::string Output string
     */
    inline std::string to_string_csv() const
    {
        return std::to_string(threads) + "," + std::to_string(blocks);
    }
};

/**
 * @brief Stream output operator for HyperParameterStorage struct
 *
 * @param os
 * @param hp
 * @return std::ostream&
 */
inline std::ostream& operator<<(std::ostream& os, const HyperParameterStorage& hp)
{
    os << "threads: " << hp.threads << " blocks: " << hp.blocks;
    return os;
}



/**
 * @brief Dummy struct for no options into Hyperparameters
 *
 */
struct NoOptions
{
};


/**
 * @brief Struct to store and use various hyperparameters for Linspace
 *
 * This is particularly useful for defining a type and then passing it through
 * as a type for easy extraction of the values.
 *
 * @tparam StartThreads
 * @tparam EndThreads
 * @tparam NThreads
 * @tparam StartBlocks
 * @tparam EndBlocks
 * @tparam NBlocks
 */
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

/**
 * @brief Single Launch Bound to force Kokkos to use set blocks and threads
 *
 * @tparam MaxThreads
 * @tparam MinBlocks
 */
template<unsigned int MaxThreads, unsigned int MinBlocks>
struct SingleLaunchBound
{
    static constexpr unsigned int maxT = MaxThreads;
    static constexpr unsigned int minB = MinBlocks;
};

/**
 * @brief A wrapper struct to use as a type for Hyperparameters
 *
 * @tparam ConfigOption NoOptions, SingleLaunchBound, and LinspaceOptions are all accepted
 */
template<typename ConfigOption = NoOptions>
struct HyperparameterOptions
{
    using DevicePolicyType = ConfigOption;
};


/**
 * @brief SFINAE Function for determining if struct is Linspace Options at compile time, FALSE
 *
 * @tparam T
 */
template<typename T>
struct is_linspace_options : std::false_type
{
};


/**
 * @brief SFINAE Function for determining if struct is Linspace Options at compile time, TRUE
 *
 * @tparam T
 */
template<unsigned int StartThreads,
         unsigned int EndThreads,
         unsigned int NThreads,
         unsigned int StartBlocks,
         unsigned int EndBlocks,
         unsigned int NBlocks>
struct is_linspace_options<
  LinspaceOptions<StartThreads, EndThreads, NThreads, StartBlocks, EndBlocks, NBlocks>>
  : std::true_type
{
};

/**
 * @brief Compile-time check for if something is a LinspaceOptions object
 *
 * @tparam T
 */
template<typename T>
inline constexpr bool is_linspace_options_v = is_linspace_options<T>::value;


/**
 * @brief SFINAE Function for determining if struct is LaunchBounds at compile time, FALSE
 *
 * @tparam T
 */
template<typename T>
struct is_launchbounds_options : std::false_type
{
};

/**
 * @brief SFINAE Function for determining if struct is LaunchBounds at compile time, TRUE
 *
 * @tparam T
 */
template<unsigned int maxT, unsigned int minB>
struct is_launchbounds_options<SingleLaunchBound<maxT, minB>> : std::true_type
{
};


/**
 * @brief Compile-time check for if something is a LaunchBounds object
 *
 * @tparam T
 */
template<typename T>
inline constexpr bool is_launchbounds_options_v = is_launchbounds_options<T>::value;


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
    // TODO: the DeviceExecutionPolicyData type can probably be inferred
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

/**
 * @brief Wrapper function that properly calls a kernel
 *
 * @tparam KernelRank
 * @tparam ViewsType
 * @tparam RangePolicyType
 * @tparam FunctorType
 * @param name Name of the kernel
 * @param range_policy Target range policy
 * @param views The views
 * @param functor Functor pointer
 */
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
    else if constexpr (KernelRank == 3)
    {
        Kokkos::parallel_for(name, range_policy, KOKKOS_LAMBDA(int i, int j, int k) {
            functor(views, i, j, k);
        });
    }
}

/**
 * @brief Wrapper function that properly calls a kernel
 *
 * @tparam KernelType
 * @param k The Kernel object
 * @param device_selector The device selector
 * @param idx The index for the hyperparameter selection
 */
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

/**
 * @brief Maps kernel input and output to a mask
 *
 * @tparam T
 * @param args
 * @return auto
 */
template<typename... T>
inline auto kernel_io_map(T&... args)
{
    return std::make_tuple(
      static_cast<bool>(std::is_const_v<std::remove_reference_t<decltype(args)>>)...);
}


/**
 * @brief Compile-time calculation of lin-space outputing integers
 *
 * @param start
 * @param end
 * @param nsteps
 * @param i
 * @return constexpr auto
 */
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

/**
 * @brief Create a range policy device object
 *
 * @tparam KernelRank
 * @tparam ExecutionSpace
 * @tparam LinspaceOptions
 * @param extent Extent of the kernel
 * @return constexpr auto
 */
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


/**
 * @brief Create a range policy device object
 *
 * @tparam KernelRank
 * @tparam ExecutionSpace
 * @param extent
 * @return constexpr auto
 */
template<int KernelRank, typename ExecutionSpace = Kokkos::KOKKOS_DEVICE>
constexpr auto create_range_policy_device(const RangeExtent<KernelRank>& extent)
{
    using DeviceRangePolicy = typename RangePolicy<KernelRank, ExecutionSpace>::type;

    // then we can create our object
    return std::make_tuple(DeviceRangePolicy(extent.lower, extent.upper));
}

// innermost function for generating the hyperparameter storage
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

/**
 * @brief Create a range policy device collection object at compile time
 *
 * @tparam LinspaceOptions
 * @return constexpr auto
 */
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


/**
 * @brief Create a range policy device collection object at compile time, empty options
 *
 * @return auto
 */
inline auto create_range_policy_device_collection()
{
    // just return a vector with one type
    return std::vector<HyperParameterStorage>({ HyperParameterStorage({ 0, 0 }) });
}

/**
 * @brief Create a standard policy object
 *
 * @tparam KernelRank
 * @tparam LaunchBoundsType
 * @tparam ExecutionSpace
 * @param extent
 * @return constexpr auto
 */
template<int KernelRank, typename LaunchBoundsType, typename ExecutionSpace = Kokkos::KOKKOS_DEVICE>
constexpr auto create_standard_policy(const RangeExtent<KernelRank>& extent)
{
    using DeviceRangePolicy = typename RangePolicy<
      KernelRank,
      ExecutionSpace,
      Kokkos::LaunchBounds<LaunchBoundsType::maxT, LaunchBoundsType::minB>>::type;

    // then we can create our object
    return std::make_tuple(DeviceRangePolicy(extent.lower, extent.upper));
}

/**
 * @brief Create the policies from hyperparameters based on the hyperparameter type
 *
 * @tparam KernelRank
 * @tparam DeviceType
 * @tparam HyperparamType
 * @param extent
 * @return constexpr auto
 */
template<int KernelRank, typename DeviceType, typename HyperparamType>
constexpr auto make_policy_from_hyperparameters(const RangeExtent<KernelRank>& extent)
{
    using BaseType = typename HyperparamType::DevicePolicyType;

    if constexpr (std::is_same_v<BaseType, NoOptions>)
    {
        return create_range_policy_device<KernelRank, DeviceType>(extent);
    }
    else if constexpr (is_linspace_options_v<BaseType>)
    {
        return create_range_policy_device<KernelRank, DeviceType, BaseType>(extent);
    }
    else if constexpr (is_launchbounds_options_v<BaseType>)
    {
        return create_standard_policy<KernelRank, BaseType>(extent);
    }
    else
    {
        throw std::runtime_error(
          "The kernel was misconfigured! Please fix the hyperparameters coming in!");
    }
}

/**
 * @brief Creates the hyperparameter vector for printing and output
 *
 * @tparam HyperparamType
 * @return constexpr auto Vector of values for HyperParameters
 */
template<typename HyperparamType>
constexpr auto make_hyperparameter_vector()
{
    using BaseType = typename HyperparamType::DevicePolicyType;

    if constexpr (std::is_same_v<BaseType, NoOptions>)
    {
        return create_range_policy_device_collection();
    }
    else if constexpr (is_linspace_options_v<BaseType>)
    {
        return create_range_policy_device_collection<BaseType>();
    }
    else if constexpr (is_launchbounds_options_v<BaseType>)
    {
        return std::vector<HyperParameterStorage>(
          { HyperParameterStorage({ BaseType::maxT, BaseType::minB }) });
    }
    else
    {
        throw std::runtime_error(
          "The kernel was misconfigured! Please fix the hyperparameters coming in!");
    }
}
