// Assumptions:
// * Algorithms are cast as operating on one element at a time
// * Kernels are steps in the algorithm that typically involve a loop over the elements
// * Inputs and outputs are all ndarrays at the element level
//

#include "Kokkos_Core.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

#include <iostream>
#define USE_EIGEN
#ifdef USE_EIGEN
#include "Eigen"
#endif

// #define NDEBUG
#ifdef NDEBUG
#include <iostream>
#endif

#ifndef KOKKOS_HOST
#define KOKKOS_HOST Serial
#endif

#ifndef KOKKOS_DEVICE
#define KOKKOS_DEVICE Serial
#endif

#define TIMING(k, f)                                 \
    {                                                \
        Kokkos::Timer timer;                         \
        timer.reset();                               \
        f;                                           \
        double kernel_time = timer.seconds();        \
        printf("%s: %.6f\n", k.name(), kernel_time); \
    }


enum class DeviceSelector {HOST, DEVICE};

//=============================================================================
// Utilities
//=============================================================================

/// @brief Packs a list of references to class instances into a tuple
/// @tparam ...ParameterTypes
/// @param ...params
/// @return
template<typename... ParameterTypes>
auto pack(ParameterTypes&... params)
{
    // note: std::forward loses the reference qualifer... check into this later
    return std::make_tuple(std::ref(params)...);
}

/*
template<typename T>
concept IsConst = std::is_const_v<std::remove_reference_t<T>>;

template<typename T>
concept NotConst = not std::is_const_v<std::remove_reference_t<T>>;
*/


// utility function for iterating over a tuple of unknown length
template<typename LambdaType, std::size_t I = 0, typename... T>
inline typename std::enable_if<I == sizeof...(T), void>::type iter_tuple(const std::tuple<T...>& t,
                                                                         const LambdaType& lambda)
{
}
template<typename LambdaType, std::size_t I = 0, typename... T>
  inline typename std::enable_if <
  I<sizeof...(T), void>::type iter_tuple(const std::tuple<T...>& t, const LambdaType& lambda)
{
    auto& elem = std::get<I>(t);
    lambda(I, elem);
    iter_tuple<LambdaType, I + 1, T...>(t, lambda);
}

//=============================================================================
// Specializations
//=============================================================================

// Concepts that will be used for EquivalentView

template<class, template<class...> class>
inline constexpr bool is_specialization = false;
template<template<class...> class T, class... Args>
inline constexpr bool is_specialization<T<Args...>, T> = true;

template<typename T>
concept IsStdVector = is_specialization<std::decay_t<T>, std::vector>;

#ifdef USE_EIGEN
template<typename T>
concept IsEigenMatrix = std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>>;
#endif

//=============================================================================
// EquivalentView
//=============================================================================

/// Used to map a container to the corresponding view type
template<typename ExecutionSpace, typename T>
struct EquivalentView;

template<typename ExecutionSpace, typename T>
    requires IsStdVector<T>
struct EquivalentView<ExecutionSpace, T>
{
    // Type of the scalar in the data structure
    using value_type =
      std::conditional_t<std::is_const_v<T>,
                         std::add_const_t<typename std::remove_reference_t<T>::value_type>,
                         typename std::remove_reference_t<T>::value_type>;

    // Type for the equivalent view of the data structure
    using type = Kokkos::View<value_type*,
                              typename ExecutionSpace::array_layout,
                              typename ExecutionSpace::memory_space>;
};

#ifdef USE_EIGEN
template<typename EigenT, typename KokkosLayout>
concept IsLayoutSame =
  (std::is_same_v<KokkosLayout, Kokkos::LayoutRight> && std::decay_t<EigenT>::IsRowMajor == 1) ||
  (std::is_same_v<KokkosLayout, Kokkos::LayoutLeft> && std::decay_t<EigenT>::IsRowMajor == 0);
#endif

#ifdef USE_EIGEN
template<typename ExecutionSpace, typename T>
    requires IsEigenMatrix<T>
struct EquivalentView<ExecutionSpace, T>
{
    // Type of the scalar in the data structure
    using value_type = std::conditional_t<std::is_const_v<T>,
                                          const typename std::remove_reference_t<T>::value_type,
                                          typename std::remove_reference_t<T>::value_type>;

    // Type for the equivalent view of the data structure
    using type = Kokkos::View<value_type**,
                              typename ExecutionSpace::array_layout,
                              typename ExecutionSpace::memory_space>;
};
#endif

//=============================================================================
// Views
//=============================================================================

template<typename ExecutionSpace>
struct Views
{
    // Create a view for a given executation space and C++ data structure
    // (each structure needs a specialization)
    template<typename T>
    static typename EquivalentView<ExecutionSpace, T>::type create_view(T&);

    // Specialization for std::vector (default allocator)
    template<typename T>
        requires IsStdVector<T>
    static auto create_view(T& vector)
    {
        return typename EquivalentView<ExecutionSpace, T>::type(vector.data(), vector.size());
    }

#ifdef USE_EIGEN
    // Specialization for Eigen matrix
    template<typename T>
        requires IsEigenMatrix<T>
    static auto create_view(T& matrix)
    {
        using ViewType = typename EquivalentView<ExecutionSpace, T>::type;
        return ViewType(matrix.data(), matrix.rows(), matrix.cols());
    }
#endif

    // Creates view for a given execution space for a variadic list of data structures
    // (each needs a create_view specialization)
    template<typename... ParameterTypes>
    static auto create_views(ParameterTypes&&... params)
    {
        return std::make_tuple(Views<ExecutionSpace>::create_view(params)...);
    }

    template<typename Tuple, std::size_t... I>
    static auto create_views_helper(const Tuple& params_tuple,
                                    std::integer_sequence<std::size_t, I...>)
    {
        return std::make_tuple(Views<ExecutionSpace>::create_view(std::get<I>(params_tuple))...);
    }

    template<typename... ParameterTypes>
    static auto create_views_from_tuple(std::tuple<ParameterTypes...> params_tuple)
    {
        return create_views_helper(params_tuple,
                                   std::make_index_sequence<sizeof...(ParameterTypes)> {});
    }
};


//=============================================================================
// RangePolicy
//=============================================================================

// Used to rank to the corresponding RangePolicy type
template<int KernelRank, typename ExecutionSpace>
struct RangePolicy;

// 1-dimensional ranges (MDRangePolicy does not support 1D ranges)
template<typename ExecutionSpace>
struct RangePolicy<1, ExecutionSpace>
{
    using type = Kokkos::RangePolicy<ExecutionSpace>;
};

// Dimensions > 1
template<int KernelRank, typename ExecutionSpace>
struct RangePolicy
{
    using type = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<KernelRank>>;
};


//=============================================================================
// RangeExtent
//=============================================================================

using ArrayIndex = std::uint64_t;

// Used to map rank to the corresponding extent for ranges type
template<int KernelRank>
struct RangeExtent;

// 1-dimensional ranges (MDRangePolicy does not support 1D ranges)
template<>
struct RangeExtent<1>
{
    using value_type = ArrayIndex;
    value_type lower;
    value_type upper;
};

// Dimensions > 1
template<int KernelRank>
struct RangeExtent
{
    using value_type = Kokkos::Array<ArrayIndex, KernelRank>;
    value_type lower;
    value_type upper;
};


RangeExtent<1> range_extent(const ArrayIndex& lower, const ArrayIndex& upper)
{
    return { lower, upper };
}

RangeExtent<2> range_extent(const Kokkos::Array<ArrayIndex, 2>& lower,
                            const Kokkos::Array<ArrayIndex, 2>& upper)
{
    return { lower, upper };
}


//=============================================================================
// Kernel
//=============================================================================

template<int KernelRank, class FunctorType, typename... ParameterTypes>
class Kernel
{
  public:
    static constexpr int rank  = KernelRank;
    
    // Note: we are choosing the host and device excution space at compile time
    using HostExecutionSpace   = Kokkos::KOKKOS_HOST;
    using DeviceExecutionSpace = Kokkos::KOKKOS_DEVICE;
    using BoundType            = RangeExtent<KernelRank>::value_type;
    using DataTuple            = std::tuple<ParameterTypes&...>;
    using HostRangePolicy      = typename RangePolicy<KernelRank, HostExecutionSpace>::type;
    using DeviceRangePolicy    = typename RangePolicy<KernelRank, DeviceExecutionSpace>::type;
    using DataParamsType       = std::tuple<ParameterTypes&...>;
    using HostDataViewsType    =
        std::tuple<typename EquivalentView<HostExecutionSpace, ParameterTypes>::type...>;
    using DeviceDataViewsType  =
        std::tuple<typename EquivalentView<DeviceExecutionSpace, ParameterTypes>::type...>;

    Kernel(const char* name,
                  std::tuple<ParameterTypes&...> params,
                  const RangeExtent<KernelRank>& extent)
      : kernel_name_(std::string(name))
      , data_params_(params)
      , data_views_host_(Views<HostExecutionSpace>::create_views_from_tuple(params))
      , data_views_device_(Views<DeviceExecutionSpace>::create_views_from_tuple(params))
      , range_lower_(extent.lower)
      , range_upper_(extent.upper)
      , range_policy_host_(HostRangePolicy(extent.lower, extent.upper))
      , range_policy_device_(DeviceRangePolicy(extent.lower, extent.upper))
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
        
    // Note: TODO think about how to make range policy generic to work for:
    // 1) Matrix-vector operation (different range)
    // 2) Matrix-matrix operation
    // 3) Vector-vector operation of same size
    // 4) vector-vector opertion with difference sizes (convolution)

    // kernel name for debugging
    std::string         kernel_name_;

    // The kernel code that will be called in an executation on the respective views
    //HostFunctorType     kernel_functor_host_;
    //DeviceFunctorType   kernel_functor_device_;
    FunctorType         kernel_functor_;

    // Data parameters and views thereof (in the execution spaces that will be considered)
    DataParamsType      data_params_;
    HostDataViewsType   data_views_host_;
    DeviceDataViewsType data_views_device_;

    // Properties pertaining to range policy
    const BoundType     range_lower_;
    const BoundType     range_upper_;
    // tile_type tile_;
    HostRangePolicy     range_policy_host_;
    DeviceRangePolicy   range_policy_device_;
};

//=============================================================================
// Data Graph
//=============================================================================

/*
// Source: https://www.fluentcpp.com/2021/03/05/stdindex_sequence-and-its-improvement-in-c20/
template<class Tuple, class F>
constexpr decltype(auto) for_each_tuple_w_index(Tuple&& tuple, F&& f)
{
    return []<std::size_t... I>(Tuple&& tuple, F&& f, std::index_sequence<I...>)
    {
        (f(std::get<I>(tuple)), ...);
        return f;
    }(std::forward<Tuple>(tuple),
      std::forward<F>(f),
      std::make_index_sequence<std::tuple_size<std::remove_reference_t<Tuple>>::value> {});
}

struct DataParamRef
{
    std::size_t kernel_index;
    std::size_t data_param_index;

    constexpr bool operator==(const DataParamRef& other)
    {
        return this->kernel_index == other.kernel_index &&
               this->data_param_index == other.data_param_index;
    }
};

// Helper function to find the smallest value in a tuple
template<std::size_t I, typename DataParamRefTuple>
constexpr auto find_min_L_impl(const DataParamRefTuple& tuple,
                               DataParamRef current_min = {
                                 std::numeric_limits<std::size_t>::max(),
                                 std::numeric_limits<std::size_t>::max() })
{
    if constexpr (I == std::tuple_size_v<DataParamRefTuple>n
    {
        return current_min;
    }
    else
    {
        const auto current_value = std::get<I>(tuple);
        const auto min_value     = current_value.data_param_index < current_min.data_param_index
                                     ? current_value
                                     : current_min;
        return find_min_L_impl<I + 1>(tuple, min_value);
    }
}

template<typename DataParamRefTuple>
constexpr DataParamRef find_min_L(const DataParamRefTuple& tuple)
{
    return find_min_L_impl<0>(tuple);
}

template<std::size_t I, typename DataParamRefTuple>
constexpr auto find_max_K_impl(const DataParamRefTuple& tuple,
                               DataParamRef current_max = {
                                 std::numeric_limits<std::size_t>::min(),
                                 std::numeric_limits<std::size_t>::min() })
{
    if constexpr (I == std::tuple_size_v<DataParamRefTuple>)
    {
        return current_max;
    }
    else
    {
        const auto current_value = std::get<I>(tuple);
        const auto max_value =
          current_value.kernel_index > current_value.kernel_index &&
              current_value.data_param_index != std::numeric_limits<std::size_t>::min()
            ? current_value
            : current_value;
        return find_max_K_impl<I + 1>(tuple, max_value);
    }
}

template<typename DataParamRefTuple>
constexpr DataParamRef find_max_K(const DataParamRefTuple& tuple)
{
    return find_max_K_impl<0>(tuple);
}

template<typename KernelsTuple, std::size_t I, std::size_t J, std::size_t K, std::size_t... L>
constexpr auto match_input_IJK_over_L(std::integer_sequence<std::size_t, L...>)
{
    using this_kernel_data_tuple = std::decay_t<std::tuple_element_t<I, KernelsTuple>>::DataTuple;
    using compared_kernel_data_tuple =
      std::decay_t<std::tuple_element_t<K, KernelsTuple>>::DataTuple;

    return std::make_tuple(
      [&]() -> DataParamRef
      {
          if constexpr (DoTagsMatch<this_kernel_data_tuple, J, compared_kernel_data_tuple, L>)
          {
              return { K, L };
          }
          else
          {
              return { K, std::numeric_limits<std::size_t>::max() };
          }
      }()...);
}

// I - index of kernel in KernelsTuple
// J - data parameter index in the Ith kernel
// K - index of another kernel in KernelsTuple to search its data
template<typename KernelsTuple, std::size_t I, std::size_t J, std::size_t... K>
constexpr auto match_input_IJ_over_K(std::integer_sequence<std::size_t, K...>)
{
    // static_assert(std::is_const_v<std::remove_reference_t<decltype(std::get<J>(
    //                 std::get<I>(algorithm_kernels).data_params_))>>,
    //              "kernel[I].data[J] is not an input (it is not const)");

    using this_kernel_data_tuple = std::decay_t<std::tuple_element_t<I, KernelsTuple>>::DataTuple;

    // Loop over the data parameters in kernel[K] and look for a match to kernel[I]->data[J]
    // that is also not const (an output)
    return std::make_tuple(find_min_L(match_input_IJK_over_L<KernelsTuple, I, J, K>(
      std::make_index_sequence<std::tuple_size_v<this_kernel_data_tuple>> {}))...);
}

template<typename KernelsTuple, std::size_t I, std::size_t... J>
constexpr auto match_input_I_over_J(std::integer_sequence<std::size_t, J...>)
{
    using this_kernel_data_tuple = std::decay_t<std::tuple_element_t<I, KernelsTuple>>::DataTuple;

    return std::make_tuple(
      find_max_K(match_input_IJ_over_K<KernelsTuple, I, J>(std::make_index_sequence<I> {}))...);
}


template<typename KernelsTuple, std::size_t I>
constexpr auto match_input()
{
    using this_kernel_data_tuple = std::decay_t<std::tuple_element_t<I, KernelsTuple>>::DataTuple;

    return match_input_I_over_J<KernelsTuple, I>(
      std::make_index_sequence<std::tuple_size_v<this_kernel_data_tuple>> {});
}

// Find dependencies for a single
// I - index of kernel in KernelsTuple
// J - data parameter index in the Ith kernel
template<typename KernelsTuple, std::size_t I, std::size_t J>
constexpr DataParamRef find_input_depencies_helper(const KernelsTuple& algorithm_kernels)
{
    // Loop over algorithm kernels from 0 to I - 1 to find any output data parameter in those
    // kernels that matches the address for kernel[I]->data[J]
}

template<typename KernelsTuple, std::size_t I>
constexpr std::tuple<DataParamRef> find_input_depencies(const KernelsTuple& algorithm_kernels)
{
    // for_each(algorithm_kernels, []())
}

*/

//=============================================================================
// Algorithm
//=============================================================================

/*
// Serves as a "node" in the data/kernel dependency graph
struct DataGraphNode
{
    DataGraphNode(const size_t I)
      : kernel_id(I)
    {
    }

    // The kernel that this node refers to in the chain of kernels
    const size_t I;

    // For each input data parameter for the corresponding kernel, this
    // indicates the upstream kernel index that outputs the data parameter needed by this kernel
    // along with the index of the data parameter in that upstream kernel.
    std::vector<std::tuple<size_t, size_t, size_t>> inputs = {};

    // For each output data parameter for the corresponding kernel, this
    // indicates the downstream kernel indices that need the data parameter as an input along
    // with the index of the data parameter in those downstream kernels.
    std::vector<std::tuple<size_t, size_t, size_t>> outputs = {};

};
*/

// build the DataGraph from chain of Kernels
// template<typename TupleType>
// auto build_data_graph (TupleType& kernels)
template<typename... KernelTypes>
auto build_data_graph(std::tuple<KernelTypes&...> kernels)
{
    using index_pair = std::tuple<size_t, size_t>;
    using index_map  = std::map<index_pair, index_pair>;
    size_t null_v    = std::numeric_limits<std::size_t>::max();

    // create an empty inputs and outputs for each
    index_map inputs;
    index_map outputs;

    // left kernel
    iter_tuple(
      kernels,
      [&]<typename KernelTypeL>(size_t il, KernelTypeL& kernel_l)
      {
          // left data param
          iter_tuple(
            kernel_l.data_params_,
            [&]<typename ParamTypeL>(size_t jl, ParamTypeL& param_l)
            {
                bool is_const_l = std::is_const_v<std::remove_reference_t<decltype(param_l)>>;

                // right kernel
                bool is_match = false;
                iter_tuple(
                  kernels,
                  [&]<typename KernelTypeJ>(size_t ir, KernelTypeJ& kernel_r)
                  {
                      if (ir <= il)
                          return;

                      // right data param
                      iter_tuple(
                        kernel_r.data_params_,
                        [&]<typename ParamTypeR>(size_t jr, ParamTypeR& param_r)
                        {
                            bool is_const_r =
                              std::is_const_v<std::remove_reference_t<decltype(param_r)>>;
                            // printf("(%d %d %d) (%d %d %d)\n", (int) il, (int) jl, (is_const_l) ?
                            // 1 : 0, (int) ir, (int) jr, (is_const_r) ? 1 : 0);

                            // match
                            if ((&param_l == &param_r) && (!is_const_l) && (is_const_r))
                            {
                                // printf("param %d in kernel %d depends on param %d in kernel
                                // %d\n",
                                //   (int) jr, (int) ir, (int) jl, (int) il);
                                outputs.emplace(std::make_tuple(il, jl), std::make_tuple(ir, jr));
                                inputs.emplace(std::make_tuple(ir, jr), std::make_tuple(il, jl));
                                is_match = true;
                                return;
                            }
                        }); // end jr

                      // found a match for this data param
                      if (is_match)
                          return;
                  }); // end ir

                // found a match for this data param
                if (is_match)
                    return;

                // if entry wasn't added yet, map it to null
                if (is_const_l)
                { // input
                    inputs.emplace(std::make_tuple(il, jl), std::make_tuple(null_v, null_v));
                }
                else
                { // output
                    outputs.emplace(std::make_tuple(il, jl), std::make_tuple(null_v, null_v));
                }
            }); // end jl
      });       // end il

#ifdef NDEBUG
    printf("\ninputs\n");
    for (const auto& item : inputs)
    {
        const auto& key   = item.first;
        const auto& value = item.second;
        std::cout << "Key: (" << std::get<0>(key) << ", " << std::get<1>(key) << "), Value: ("
                  << std::get<0>(value) << ", " << std::get<1>(value) << ")\n";
    }
    printf("\noutputs\n");
    for (const auto& item : outputs)
    {
        const auto& key   = item.first;
        const auto& value = item.second;
        std::cout << "Key: (" << std::get<0>(key) << ", " << std::get<1>(key) << "), Value: ("
                  << std::get<0>(value) << ", " << std::get<1>(value) << ")\n";
    }
#endif

    /*
      // now we have maps of all data param connections!

      // next, loop over kernels and make a node for each kernel
      //   how to determine number of inputs and outputs for each kernel?
      //   should I have counted them?
      //   or should we just use vectors?

      // should outputs will null destinations be automatically copied back to the host?
    */
}

// main algorithm object

template<typename... KernelTypes>
class Algorithm
{
  public:
    // constructor should initialize and empty vector
    constexpr Algorithm(std::tuple<KernelTypes&...> kernels)
      : kernels_(kernels)
    {
#ifdef NDEBUG
        iter_tuple(kernels_,
                   []<typename KernelType>(size_t i, KernelType& kernel)
                   { printf("Registered Kernel: %s\n", kernel.kernel_name_.c_str()); });
#endif
    };
    ~Algorithm() {};

    // the core of this class is a tuple of kernels
    std::tuple<KernelTypes&...> kernels_;

    /*
    // call all kernels
    void call()
    {
        iter_tuple(kernels_,
                   []<typename KernelType>(size_t i, KernelType& kernel)
                   { TIMING(kernel, kernel.call()); });
    };
    */
};

/*
template<int KernelRank, typename FunctorType, typename ViewsType>
void functor_lambda(const FunctorType functor, ViewsType& views) {
    return;
}

template<typename FunctorType, typename ViewsType>
std::function<void (int)> functor_lambda<1>(const FunctorType functor, ViewsType& views) {
    return KOKKOS_LAMBDA(int i) { functor(views, i); };
}

template<typename FunctorType, typename ViewsType>
std::function<void (int, int)> functor_lambda<2>(const FunctorType functor, ViewsType& views) {
    return KOKKOS_LAMBDA(int i, int j) { functor(views, i, j); };
}
*/

template<int KernelRank, typename ViewsType, typename RangePolicyType, typename FunctorType>
void call_kernel(const std::string& name,
                 const RangePolicyType& range_policy,
                 ViewsType& views,
                 const FunctorType functor)
{
    //Kokkos::parallel_for(name, range_policy, functor_lambda<KernelRank>(functor, views));
    if constexpr (KernelRank == 1) {
        Kokkos::parallel_for(name, range_policy,
            KOKKOS_LAMBDA(int i) { functor(views, i); });
    } else if constexpr (KernelRank == 2) {
        Kokkos::parallel_for(name, range_policy,
            KOKKOS_LAMBDA(int i, int j) { functor(views, i, j); });
    }
}

template<typename KernelType>
void call_kernel(KernelType k, DeviceSelector device_selector)
{
    if (device_selector == DeviceSelector::HOST)
    {
        call_kernel<KernelType::rank>(k.kernel_name_,
                                      k.range_policy_host_,
                                      k.data_views_host_,
                                      k.kernel_functor_);
    }
    else if (device_selector == DeviceSelector::DEVICE)
    {
        call_kernel<KernelType::rank>(k.kernel_name_,
                                      k.range_policy_device_,
                                      k.data_views_device_,
                                      k.kernel_functor_);
    }
}


//=============================================================================
// Main
//=============================================================================

// degrees of freedom:
// execution space: host, device
// execution order:
/*    
struct FunctorK1 {
    template<typename ViewsTuple, typename Index>
    void operator()(ViewsTuple& views, const Index& i) const {
        auto& x = std::get<0>(views);
        auto& y = std::get<1>(views);
        auto& z = std::get<2>(views);
        z[i]    = x[i] * y[i];
    }
};
template<typename... ParameterTypes>
auto K1(ParameterTypes&... data_params)
{
  auto name   = "1D vector-vector multiply 1";
  auto params = pack(data_params...);
  auto extent = range_extent(0, std::get<2>(params).size());
  return Kernel<1, FunctorK1, ParameterTypes...>(name, params, extent);
}


struct FunctorK2 {
    template<typename ViewsTuple, typename Index>
    void operator()(ViewsTuple& views, const Index& i) const {
        auto& x = std::get<0>(views);
        auto& z = std::get<1>(views);
        auto& w = std::get<2>(views);
        w[i]    = x[i] * z[i];
    }
};
template<typename... ParameterTypes>
auto K2(ParameterTypes&... data_params)
{
  auto name   = "1D vector-vector multiply 2";
  auto params = pack(data_params...);
  auto extent = range_extent(0, std::get<2>(params).size());
  return Kernel<1, FunctorK2, ParameterTypes...>(name, params, extent);
}


struct FunctorK3 {
    template<typename ViewsTuple, typename Index>
    void operator()(ViewsTuple& views, const Index& i) const {
        auto& x = std::get<0>(views);
        auto& z = std::get<1>(views);
        auto& q = std::get<2>(views);
        q[i]    = x[i] * z[i];
    }
};
template<typename... ParameterTypes>
auto K3(ParameterTypes&... data_params)
{
  auto name   = "1D vector-vector multiply 3";
  auto params = pack(data_params...);
  auto extent = range_extent(0, std::get<2>(params).size());
  return Kernel<1, FunctorK3, ParameterTypes...>(name, params, extent);
}
*/

struct FunctorK4 {
    template<typename ViewsTuple, typename Index>
    void operator()(ViewsTuple& views, const Index& i, const Index& j) const {
        auto& A = std::get<0>(views);
        auto& x = std::get<1>(views);
        auto& b = std::get<2>(views);
        b(i) += A(i,j) * x(j);
    }
};
template<typename... ParameterTypes>
auto K4(ParameterTypes&... data_params)
{
  auto name   = "matrix-vector multiply";
  auto params = pack(data_params...);
  unsigned long N = std::get<0>(params).rows();
  unsigned long M = std::get<0>(params).cols();
  auto extent = range_extent({ 0, 0 }, { N, M });
  return Kernel<2, FunctorK4, ParameterTypes...>(name, params, extent);
}

//=============================================================================
// main
//=============================================================================

int main(int argc, char* argv[])
{
    int N;

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    N = 5;
    std::vector<double> x(N);
    std::iota(x.begin(), x.end(), 0.0);
    std::vector<double> y(N);
    std::iota(y.begin(), y.end(), 0.0);
    std::vector<double> z(N);
    std::vector<double> w(N);
    std::vector<double> q(N);

//    auto k1 = K1(std::as_const(x), std::as_const(y), z); // vvm
//    auto k2 = K2(std::as_const(x), std::as_const(z), w); // vvm
    //auto k3 = K3(std::as_const(x), std::as_const(z), q); // vvm

    // Create an Algorithm object
    //Algorithm algo(pack(k1, k2, k3));
    //build_data_graph(algo.kernels_);

        
    Eigen::MatrixXd a(N, N);
    //a.setRandom();
    a.setIdentity();
    std::vector<double> b(N, 2.0);
    std::vector<double> c(N, 0.0);
    
    auto k4 = K4(std::as_const(a), std::as_const(b), c); // mvm
    
    /*
    // matrix-vector multiply
    {
        // set up data
        int N = 10000000;
        std::vector<double> x(N);
        std::iota(x.begin(), x.end(), 0.0);
        Eigen::MatrixXd y(N, N);
        y.setRandom();
        std::vector<double> z(N);

        // define the kernel
        Kernel k(
            "matrix-vector multiply",
            pack(std::as_const(x), std::as_const(y), z),
            []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i, const
    Index& j)
            {
                auto& x = std::get<0>(views);
                auto& y = std::get<1>(views);
                auto& z = std::get<2>(views);
                z[i] += y[i, j] * x[j];
            },
            range_extent({ 0, 0 }, { N, N })
        );

        // run the kernel
        TIMING(k.call());

        // verify the output
        for (int i = 0; i < z.size(); i++) {
          double tmp = 0;
          for (int j = 0; j < z.size(); i++) {
            tmp = y[i, j] * x[i];
          }
          assert(x[i] * x[i] == z[i]);
        }

    }
    */

    /*
    // 2D convolution
    {
        // set up data
        unsigned int N = 32;
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
    Eigen::RowMajor>;

        MatrixType x(N, N); // 32x32
        x.setRandom();
        N -= 2;
        MatrixType y(N, N); // 30x30
        y.setZero();

        Kernel k(
          "2D convolution",
          pack(std::as_const(x), y),
          []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i, const
    Index& j)
          {
              auto& x_view = std::get<0>(views);
              auto x_subview =
                Kokkos::subview(x_view, Kokkos::make_pair(i, i + 3), Kokkos::make_pair(j, j +
    3)); auto& y_view = std::get<1>(views);

              auto tmp = 0.0;
              for (Index ii = 0; ii < 3; ii++)
              {
                  for (Index jj = 0; jj < 3; jj++)
                  {
                      tmp += x_subview(ii, jj);
                  }
              }
              y_view(i, j) = tmp;
          },
          range_extent({ 0, 0 }, { N, N }));

        k.call();

        for (auto i = 0; i < N; i++)
        {
            for (auto j = 0; j < N; j++)
            {
                auto diff = y(i, j) - x.block(i, j, 3, 3).sum();
                assert(abs(diff) < 1e-14);
            }
        }
    }
    */

    // At the end, the algorithm needs to know the "final" output that needs copied to the host
    // Data needs moved if 1) it is a kernel input or 2) algorithm output
    // Data view deallocation if 1) it is not a downstream input 2) and not algorithm output
    // - perhaps use counter for each view (+1 for algorithm output) to know when to deallocate
    // it Need algorithm to construct the counters, for example:
    //   k.parameters[1] is not const and hence output
    //   k2.parameters[0] is const and hence input
    // assert(&std::get<1>(k.parameters) == &std::get<0>(k2.parameters));

//    // TEST
    DeviceSelector device = DeviceSelector::DEVICE;
//    printf("\nk1\n");
//    k1(device);
//    for (auto i = 0; i < y.size(); i++)
//    {
//        //assert(x[i] * y[i] == z[i]);
//        printf("%f * %f = %f\n", x[i], y[i], z[i]);
//    }
//    printf("\nk2\n");
//    k2(device);
//    for (auto i = 0; i < w.size(); i++)
//    {
//        //assert(x[i] * z[i] == w[i]);
//        printf("%f * %f = %f\n", x[i], z[i], w[i]);
//    }
//    //printf("\nk3\n");
//    //k3(device);
//    //for (auto i = 0; i < q.size(); i++)
//    //{
//    //    //assert(x[i] * z[i] == q[i]);
//    //    printf("%f * %f = %f\n", x[i], z[i], q[i]);
//    //}
    printf("\nk4\n");
    k4(device);
    std::cout << "A" << std::endl;
    std::cout << a << std::endl;
    std::cout << "x" << std::endl;
    for (const double& val : b)
      std::cout << " " << val << std::endl;
    std::cout << "b" << std::endl;
    for (const double& val : c)
      std::cout << " " << val << std::endl;

    // Finalize Kokkos
    Kokkos::finalize();

    printf("\n** GRACEFUL EXIT **\n");
}
