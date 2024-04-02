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

#ifdef USE_EIGEN
#include "Eigen"
#endif

//#define NDEBUG
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


enum class DeviceSelector { HOST, DEVICE };
/*
KOKKOS_FUNCTION void test_kokkos()
{
  int N = 100;
  Kokkos::View<double*, Kokkos::Cuda> a("A", N);
  Kokkos::parallel_for("FillA", N, KOKKOS_LAMBDA(const int i) {
    a(i) = i;
  });
}
*/

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

template<int KernelRank, typename LambdaType, typename... ParameterTypes>
class Kernel
{
  public:
    // Note: we are choosing the host and device excution space at compile time
    using HostExecutionSpace   = Kokkos::KOKKOS_HOST;
    using DeviceExecutionSpace = Kokkos::KOKKOS_DEVICE;
    using BoundType            = RangeExtent<KernelRank>::value_type;
    using DataTuple            = std::tuple<ParameterTypes&...>;
    using HostRangePolicy      = typename RangePolicy<KernelRank, HostExecutionSpace>::type;
    using DeviceRangePolicy    = typename RangePolicy<KernelRank, DeviceExecutionSpace>::type;

    Kernel(const char* name,
           std::tuple<ParameterTypes&...> params,
           const LambdaType& lambda,
           const RangeExtent<KernelRank>& range_extent)
      : kernel_name_(std::string(name))
      , kernel_rank_(KernelRank)
      , kernel_lambda_(lambda)
      , data_params_(params)
      , data_views_host_(Views<HostExecutionSpace>::create_views_from_tuple(params))
      , data_views_device_(Views<DeviceExecutionSpace>::create_views_from_tuple(params))
      , range_lower_(range_extent.lower)
      , range_upper_(range_extent.upper)
      , range_policy_host_(HostRangePolicy(range_extent.lower, range_extent.upper))
      , range_policy_device_(DeviceRangePolicy(range_extent.lower, range_extent.upper))
    {
#ifdef NDEBUG
        // debugging diagnostics
        printf("\nHost Execution Space:\n");
        HostExecutionSpace {}.print_configuration(std::cout);
        printf("\nDevice Execution Space:\n");
        DeviceExecutionSpace {}.print_configuration(std::cout);
#endif
    }

    /*
    //copy constructor
    //are all these members really trivially copyable?
    Kernel(Kernel& k)
    {
      kernel_name_         = k.kernel_name_;
      kernel_rank_         = k.kernel_rank_;
      kernel_lambda_       = k.kernel_lambda_;
      data_params_         = k.data_params_;
      data_views_host_     = k.data_views_host_;
      data_views_device_   = k.data_views_device_;
      range_lower_         = k.range_lower_;
      range_upper_         = k.range_upper_;
      range_policy_host_   = k.range_policy_host_;
      range_policy_device_ = k.range_policy_device_;
    }
    */

    //virtual void call()
    void operator()(DeviceSelector device_selector)
    //virtual void call(DeviceSelector selector)
    //void call()
    {
        call_kernel(*this, device_selector);
   
        //test_kokkos();
        //{
        //  //int N = 100;
        //  //int* a_array = new a [int];
        //  //Kokkos::View<int*, Kokkos::Cuda> a("A", N);
        //  Kokkos::parallel_for(
        //    "FillA",
        //    100,
        //    KOKKOS_LAMBDA(const int i) {
        //      //a(i) = i;
        //    });
        //}

        // data movement needs to happen at the algorithm level
        //
        // Inside Kernel for the wrapper
        // TODO deep copies (somewhere)

      //  if (selector == DeviceSelector::HOST) {
      //    auto kernel_wrapper = [=, this](const auto... indices)
      //    {
      //        kernel_lambda_(data_views_host_, indices...);
      //    };

      //    using RangePolicyType = typename RangePolicy<KernelRank, HostExecutionSpace>::type;
      //    auto range_policy     = RangePolicyType(lower_, upper_);

      //    Kokkos::parallel_for("Loop", range_policy, kernel_wrapper);
      //  
      //  } else if (selector == DeviceSelector::DEVICE) {
    
        
          //using DataViewsType = std::tuple<typename EquivalentView<DeviceExecutionSpace, ParameterTypes>::type...>;

          //LambdaType kernel_lambda = kernel_lambda_;
          //DataViewsType data_views = data_views_device_;


          //using RangePolicyType = typename RangePolicy<KernelRank, DeviceExecutionSpace>::type;
          //auto range_policy     = RangePolicyType(lower_, upper_);

          //Kokkos::parallel_for("Loop", range_policy, KOKKOS_LAMBDA(const auto... indices) {
              //kernel_lambda(data_views, indices...);
          //});
      //  }

        // Note: TODO think about how to make range policy generic to work for:
        // 1) Matrix-vector operation (different range)
        // 2) Matrix-matrix operation
        // 3) Vector-vector operation of same size
        // 4) vector-vector opertion with difference sizes (convolution)
    };

    char* name()
    {
        return (char*)kernel_name_.c_str();
    }

    // kernel name for debugging
    std::string kernel_name_;

    // kernel rank for building range policies later
    const int kernel_rank_;
    
    // The kernel code that will be called in an executation on the respective views
    LambdaType kernel_lambda_;

    // Data parameters and views thereof (in the execution spaces that will be considered)
    std::tuple<ParameterTypes&...> data_params_;
    std::tuple<typename EquivalentView<HostExecutionSpace, ParameterTypes>::type...>
      data_views_host_;
    std::tuple<typename EquivalentView<DeviceExecutionSpace, ParameterTypes>::type...>
      data_views_device_;

    // Properties pertaining to range policy
    const BoundType range_lower_;
    const BoundType range_upper_;
    // tile_type tile_;
    HostRangePolicy range_policy_host_;
    DeviceRangePolicy range_policy_device_;
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
//template<typename TupleType>
//auto build_data_graph (TupleType& kernels)
template<typename... KernelTypes>
auto build_data_graph (std::tuple<KernelTypes&...> kernels)
{
  using index_pair = std::tuple<size_t, size_t>;
  using index_map  = std::map<index_pair, index_pair>;
  size_t null_v    = std::numeric_limits<std::size_t>::max();
  
  // create an empty inputs and outputs for each
  index_map inputs;
  index_map outputs;
    
  // left kernel
  iter_tuple(kernels, [&]<typename KernelTypeL>(size_t il, KernelTypeL& kernel_l) {
    
    // left data param
    iter_tuple(kernel_l.data_params_, [&]<typename ParamTypeL>(size_t jl, ParamTypeL& param_l) {
      bool is_const_l = std::is_const_v<std::remove_reference_t<decltype(param_l)>>;
    
      // right kernel
      bool is_match = false;
      iter_tuple(kernels, [&]<typename KernelTypeJ>(size_t ir, KernelTypeJ& kernel_r) {
        if (ir <= il) return;

        // right data param
        iter_tuple(kernel_r.data_params_, [&]<typename ParamTypeR>(size_t jr, ParamTypeR& param_r) {
          bool is_const_r = std::is_const_v<std::remove_reference_t<decltype(param_r)>>;
          //printf("(%d %d %d) (%d %d %d)\n", (int) il, (int) jl, (is_const_l) ? 1 : 0, (int) ir, (int) jr, (is_const_r) ? 1 : 0);
          
          // match
          if ((&param_l == &param_r) && (!is_const_l) && (is_const_r)) {
            //printf("param %d in kernel %d depends on param %d in kernel %d\n",
            //  (int) jr, (int) ir, (int) jl, (int) il);
            outputs.emplace(std::make_tuple(il,jl), std::make_tuple(ir,jr));
            inputs.emplace(std::make_tuple(ir,jr), std::make_tuple(il,jl));
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
      if (is_const_l) { //input
        inputs.emplace(std::make_tuple(il,jl), std::make_tuple(null_v, null_v));
      } else { // output
        outputs.emplace(std::make_tuple(il,jl), std::make_tuple(null_v, null_v));
      }
    
    }); // end jl

  }); // end il

  #ifdef NDEBUG
    printf("\ninputs\n");
    for (const auto& item : inputs) {
      const auto& key = item.first;
      const auto& value = item.second;
      std::cout << "Key: (" << std::get<0>(key) << ", " << std::get<1>(key)
                << "), Value: (" << std::get<0>(value) << ", " << std::get<1>(value) << ")\n";
    }
    printf("\noutputs\n");
    for (const auto& item : outputs) {
      const auto& key = item.first;
      const auto& value = item.second;
      std::cout << "Key: (" << std::get<0>(key) << ", " << std::get<1>(key)
                << "), Value: (" << std::get<0>(value) << ", " << std::get<1>(value) << ")\n";
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

/*
template <int Rank, typename... T>
class Test
{
  public:
  Test() {};

  void test()
  {
    Kokkos::parallel_for("test", 100, KOKKOS_LAMBDA(const int i) {});
  }
};
*/

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
        iter_tuple(kernels_, []<typename KernelType>(size_t i, KernelType& kernel) {
          printf("Registered Kernel: %s\n", kernel.kernel_name_.c_str());
        });
      #endif
    };
    ~Algorithm() {};

    // the core of this class is a tuple of kernels
    std::tuple<KernelTypes&...> kernels_;

    // call all kernels
    void call()
    {
      iter_tuple(kernels_, []<typename KernelType>(size_t i, KernelType& kernel) {
        TIMING(kernel, kernel.call());
      });
    };
};



template<typename KernelType>
void call_kernel_on_host(KernelType k)
{

    auto& kernel_name   = k.kernel_name_;
    auto& kernel_lambda = k.kernel_lambda_;
    auto& data_views    = k.data_views_host_;
   
    /*
    auto& range_lower     = k.range_lower_;
    auto& range_upper     = k.range_upper_;
    auto  KernelRank      = k.kernel_rank_;
    using ExecutionSpace  = Kokkos::KOKKOS_HOST;
    using RangePolicyType = typename RangePolicy<KernelRank, ExecutionSpace>::type;
    auto  range_policy    = RangePolicyType(range_lower, range_upper);
    */
    
    auto& range_policy = k.range_policy_host_;

    auto kernel_wrapper = [=](const auto... indices)
    {
        kernel_lambda(data_views, indices...);
    };

    Kokkos::parallel_for(kernel_name, range_policy, kernel_wrapper);
}

void 

//__host__ __device__ extended lambdas cannot be generic lambdas
// in other words, can only use templated lambdas with one or the other
template<typename KernelType>
void call_kernel_on_device(KernelType& k)
//void call_kernel_on_device(Kernel& k)
{

   auto& kernel_name    = k.kernel_name_;
   auto& kernel_rank    = k.kernel_rank_;
   auto& kernel_lambda  = k.kernel_lambda_;
   auto& data_views     = k.data_views_device_;
   auto& range_policy   = k.range_policy_device_;

   if (kernel_rank == 1) {
       Kokkos::parallel_for(kernel_name, range_policy, KOKKOS_LAMBDA(int i) {
         kernel_lambda(data_views, i);
       });
   } //else if (kernel_rank == 2) {} //etc.
   
}

template<typename KernelType>
void call_kernel(KernelType k, DeviceSelector device_selector)
{
    if (device_selector == DeviceSelector::HOST) {
        call_kernel_on_host(k);
    } else if (device_selector == DeviceSelector::DEVICE) {
        call_kernel_on_device(k);
    }
}

        
//=============================================================================
// Main
//=============================================================================

// degrees of freedom:
// execution space: host, device
// execution order:

int main(int argc, char* argv[])
{
    int N;

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    //test_kokkos();
    //Test<0, int, double> b;
    //b.test();

    // 1D vector-vector multiply
    N = 5;
    std::vector<double> x(N);
    std::iota(x.begin(), x.end(), 0.0);
    std::vector<double> y(N);
    std::iota(y.begin(), y.end(), 0.0);
    std::vector<double> z(N);
    std::vector<double> w(N);
    std::vector<double> q(N);

    Kernel k1(
      "1D vector-vector multiply 1",
      pack(std::as_const(x), std::as_const(y), z),
      []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
      {
          auto& x = std::get<0>(views);
          auto& y = std::get<1>(views);
          auto& z = std::get<2>(views);
          z[i]    = x[i] * y[i];
      },
      range_extent(0, z.size()));

    /*
    Kernel k2(
      "1D vector-vector multiply 2",
      pack(std::as_const(x), std::as_const(z), w),
      []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
      {
          auto& x = std::get<0>(views);
          auto& z = std::get<1>(views);
          auto& w = std::get<2>(views);
          w[i]    = x[i] * z[i];
      },
      range_extent(0, w.size()));
    
    Kernel k3(
      "1D vector-vector multiply 3",
      pack(std::as_const(x), std::as_const(z), q),
      []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
      {
          auto& x = std::get<0>(views);
          auto& z = std::get<1>(views);
          auto& q = std::get<2>(views);
          q[i]    = x[i] * z[i];
      },
      range_extent(0, q.size()));

    // Create an Algorithm object
    Algorithm algo(pack(k1, k2, k3));
        
    build_data_graph(algo.kernels_);
    */

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

    // execute all kernels
    ////algo._deduce_dependencies();
    // algo.call();
    //k1.call(DeviceSelector::HOST);
    //k1.call();
    DeviceSelector device_selector = DeviceSelector::HOST;
    k1(device_selector);

    // 1D vector-vector multiply: verify the output
    for (auto i = 0; i < y.size(); i++) {
        assert(x[i] * y[i] == z[i]);
    }

    // Finalize Kokkos
    Kokkos::finalize();

    printf("\n** GRACEFUL EXIT **\n");
}
