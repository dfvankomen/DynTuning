// Assumptions:
// * Algorithms are cast as operating on one element at a time
// * Kernels are steps in the algorithm that typically involve a loop over the elements
// * Inputs and outputs are all ndarrays at the element level
//

#include "CompileTimeCounter.h"
#include "Kokkos_Core.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

#ifdef USE_EIGEN
  #include "Eigen"
#endif

#define NDEBUG
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

template<typename T>
concept IsConst = std::is_const_v<std::remove_reference_t<T>>;

template<typename T>
concept NotConst = not std::is_const_v<std::remove_reference_t<T>>;


// utility function for iterating over a tuple of unknown length
template<typename LambdaType, std::size_t I = 0, typename... T>
inline typename std::enable_if<I == sizeof...(T), void>::type iter_tuple(
    const std::tuple<T...>& t, const LambdaType& lambda) {}
template<typename LambdaType, std::size_t I = 0, typename... T>
  inline typename std::enable_if <
  I<sizeof...(T), void>::type iter_tuple(const std::tuple<T...>& t, const LambdaType& lambda)
{
    auto& elem = std::get<I>(t);
    lambda(elem);
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
concept IsEigenMatrix = std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T> >, std::decay_t<T> >;
#endif

//=============================================================================
// Tag
//=============================================================================

// Note:
//   The issue now is that id_ is still a "runtime value", i.e. on the stack, since the class is not
//   instantiated until runtime in this case.  Instead, we need to make it part of the template
//   parameters.
//
// lightweight wrapper class to tag an object with a unique identifier
template<int TagID, typename T>
struct Tag
{
    using is_tag     = std::true_type;
    using value_type = T; // Type of underlying data

    std::remove_reference_t<T>& v_;

    static constexpr int id = TagID;

    operator T&()
    {
        return v_;
    };
};

template<int ID, typename T>
constexpr auto make_tag(T& v)
{
    return Tag<ID, T>({ v });
}

#define TAG(x) make_tag<__COUNTER__>(x);

template<typename T>
concept IsTag = std::is_same_v<typename std::decay_t<T>::is_tag, std::true_type>;

/// Checks if the Ith tag of DataTuple1 matches the Jth tag of DataTuple2
template<typename DataTuple1, int I, typename DataTuple2, int J>
concept DoTagsMatch = std::is_same_v<std::decay_t<std::tuple_element_t<I, DataTuple1>>,
                                     std::decay_t<std::tuple_element_t<J, DataTuple2>>>;

//=============================================================================
// EquivalentView
//=============================================================================

// Used to map a container to the corresponding view type

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

template<typename ExecutionSpace, typename T>
    requires IsTag<T>
struct EquivalentView<ExecutionSpace, T>
{
    using underlying_view_type = std::conditional_t<
      std::is_const_v<std::remove_reference_t<T>>,
      EquivalentView<ExecutionSpace, std::add_const_t<typename std::decay_t<T>::value_type>>,
      EquivalentView<ExecutionSpace, typename std::decay_t<T>::value_type>>;

    // Type of the scalar in the data structure
    using value_type = typename underlying_view_type::value_type;

    // Type for the equivalent view of the data structure
    using type = typename underlying_view_type::type;
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
    static typename EquivalentView<ExecutionSpace, T>::type create_view(T&&);

    // Specialization for Tag, which will call create_view for underlying type
    template<typename T>
        requires IsTag<T> && IsStdVector<T>
    static auto create_view(T& tag)
    {
        return typename EquivalentView<ExecutionSpace, T>::type(tag.v_.data(), tag.v_.size());
    }

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

    Kernel(const char* name,
           std::tuple<ParameterTypes&...> params,
           const LambdaType& lambda,
           const RangeExtent<KernelRank>& range_extent)
      : data_params_(params)
      , kernel_lambda_(lambda)
      , data_views_host_(Views<HostExecutionSpace>::create_views_from_tuple(params))
      , data_views_device_(Views<DeviceExecutionSpace>::create_views_from_tuple(params))
      , lower_(range_extent.lower)
      , upper_(range_extent.upper)
    {
        kernel_name_ = std::string(name);
#ifdef NDEBUG

        // debugging diagnostics
        printf("\nHost Execution Space:\n");
        HostExecutionSpace {}.print_configuration(std::cout);
        printf("\nDevice Execution Space:\n");
        DeviceExecutionSpace {}.print_configuration(std::cout);

#endif
    }

    virtual void call()
    {
        // data movement needs to happen at the algorithm level
        //
        // Inside Kernel for the wrapper
        // TODO deep copies (somewhere)

        auto kernel_wrapper = [=, this](const auto... indices)
        {
            kernel_lambda_(data_views_host_, indices...);
        };

        using RangePolicyType = typename RangePolicy<KernelRank, HostExecutionSpace>::type;
        auto range_policy     = RangePolicyType(lower_, upper_);

        Kokkos::parallel_for("Loop", range_policy, kernel_wrapper);

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

    // Data parameters and views thereof (in the execution spaces that will be considered)
    std::tuple<ParameterTypes&...> data_params_;
    std::tuple<typename EquivalentView<HostExecutionSpace, ParameterTypes>::type...>
      data_views_host_;
    std::tuple<typename EquivalentView<DeviceExecutionSpace, ParameterTypes>::type...>
      data_views_device_;

    // The kernel code that will be called in an executation on the respective views
    LambdaType kernel_lambda_;

    // Properties pertaining to range policy
    const BoundType lower_;
    const BoundType upper_;
    // tile_type tile_;
};


//=============================================================================
// Data Graph
//=============================================================================

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
};

template<typename KernelsTuple, std::size_t I, std::size_t J, std::size_t K, std::size_t... L>
constexpr auto match_input_data_param_helper(const KernelsTuple& algorithm_kernels,
                                             std::integer_sequence<std::size_t, L...>)
{
    auto& this_kernel     = std::get<I>(algorithm_kernels);
    auto& this_data       = std::get<J>(this_kernel.data_params_);
    auto& compared_kernel = std::get<K>(algorithm_kernels);

    static_assert(IsTag<decltype(this_data)>);

    return std::make_tuple(
      [&](auto& compared_data) -> std::size_t
      {
          static_assert(IsTag<decltype(compared_data)>);
          if constexpr (this_data.id == compared_data.id &&
                        std::is_const_v<decltype(compared_data)>)
          {
              return L;
          }
          else
          {
              return std::numeric_limits<std::size_t>::max();
          }
      }(std::get<L>(compared_kernel.data_params_))...);
}


// Helper function to find the smallest value in a tuple
template<size_t I = 0, typename... Ts>
constexpr int min_value_in_tuple(const std::tuple<Ts...>& myTuple,
                                 int currentMin = std::numeric_limits<int>::max())
{
    if constexpr (I == sizeof...(Ts))
    {
        return currentMin;
    }
    else
    {
        const int currentValue = std::get<I>(myTuple);
        return min_value_in_tuple<I + 1>(myTuple, std::min(currentMin, currentValue));
    }
}

// I - index of kernel in KernelsTuple
// J - data parameter index in the Ith kernel
// K - index of another kernel in KernelsTuple to search its data
template<std::size_t I, std::size_t J, std::size_t K, typename KernelsTuple>
constexpr std::size_t match_input_data_param(const KernelsTuple& algorithm_kernels)
{
    static_assert(std::is_const_v<std::remove_reference_t<decltype(std::get<J>(
                    std::get<I>(algorithm_kernels).data_params_))>>,
                  "kernel[I].data[J] is not an input (it is not const)");

    auto& this_kernel = std::get<I>(algorithm_kernels); // this should be the Ith Tag's value "v"
    auto& this_data   = std::get<J>(this_kernel.data_params_);
    auto& compared_kernel = std::get<K>(algorithm_kernels);

    // Loop over the data parameters in kernel[K] and look for a match to kernel[I]->data[J]
    // that is also not const (an output)
    auto matches = match_input_data_param_helper<KernelsTuple, I, J, K>(
      algorithm_kernels,
      std::make_index_sequence<std::tuple_size_v<decltype(compared_kernel.data_params_)>> {});

    return min_value_in_tuple(matches);
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

// Serves as a "node" in the data/kernel dependency graph
template<typename KernelsTuple, std::size_t I>
struct DataGraphNode
{
    constexpr DataGraphNode(const KernelsTuple& algorithm_kernels)
      : kernel_(std::get<I>(algorithm_kernels))
    {
    }

    // The kernel that this node refers to in the chain of kernels
    static constexpr std::tuple_element<I, KernelsTuple>::type& kernel_;
    // For each input data parameter for the corresponding kernel (I in the KernelsTuple), this
    // indicates the upstream kernel index that outputs the data parameter needed by this kernel
    // along with the index of the data parameter in that upstream kernel.
    static constexpr std::tuple<DataParamRef> input_dependencies_;
    // For each output data parameter for the corresponding kernel (I in the KernelsTuple), this
    // indicates the downstream kernel indices that need the data parameter as an input along
    // with the index of the data parameter in those downstream kernels.
    static constexpr std::tuple<std::tuple<DataParamRef>> output_dependencies_;
};
    
/*
// utility function to check for a data dependency
template <typename TagTypeI, typename TagTypeJ>
constexpr bool is_dependent(TagTypeI& tag1, TagTypeJ& tag2) {
    // ID match and former is not const but latter is const
    if constexpr ((tag1::id == tag2::id) && (!std::is_const(tag1)) && (std::is_const(tag2))) {
        return true;
    } else {
        return false;
    }
}
// Data params loop over J
template <std::size_t I, std::size_t J, typename KernelTypeI, typename KernelTypeJ>
constexpr void compare_data_params_IJ(KernelTypeI& k1, KernelTypeJ& k2) {
    if constexpr (is_dependent(std::get<I>(k1), std::get<J>(k2))) {
        // set up a copy operation for kernel k1 to copy to the device of k2
    }
    if constexpr (J == 0)
        return;
    else
        compare_data_params_IJ<I,J-1>(k1, k2);
}
// Data params loop over I
template <std::size_t I, typename KernelTypeI, typename KernelTypeJ>
constexpr void compare_data_params_I(KernelTypeI& k1, KernelTypeJ& k2) {
    compare_data_params_IJ<I,std::tuple_size_v<decltype(k2.data_params_)> - 1>(k1, k2);
    if constexpr (I == 0)
        return;
    else
      compare_data_params_I<I-1>(k1, k2);
}
// Init comparison of data params
template <typename KernelTypeI, typename KernelTypeJ>
constexpr void compare_data_params(KernelTypeI& k1, KernelTypeJ& k2) {
    compare_data_params_I<std::tuple_size_v<decltype(k1.data_params_)> - 1>(k1, k2);
}
// Kernels loop over I
template <std::size_t I, std::size_t J, typename... KernelsTuple>
constexpr void compare_kernels_IJ(const std::tuple<KernelsTuple...>& kernels) {
    compare_data_params(std::get<I>(kernels), std::get<J>(kernels));
    if constexpr (I == 0)
        return;
    else
        compare_kernels_IJ<I-1,J>(kernels);
}
// Kernels loop over J
template <std::size_t J, typename... KernelsTuple>
constexpr void compare_kernels_J(const std::tuple<KernelsTuple...>& kernels) {
    compare_kernels_IJ<J-1,J>(kernels);
    if constexpr (J == 0)
        return;
    else
        compare_kernels_J<J-1>(kernels);
}
// Init comparison of kernels
template <typename... KernelsTuple>
constexpr void deduce_dependencies(const std::tuple<KernelsTuple...>& kernels) {
    compare_kernels_J<std::tuple_size_v<decltype(kernels)> - 1>(kernels);
}
*/

//=============================================================================
// Algorithm
//=============================================================================

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
            iter_tuple(kernels_, []<typename KernelType>(KernelType& kernel) {
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
        iter_tuple(
            kernels_,
            []<typename KernelType>(KernelType& kernel)
            {
                TIMING(kernel, kernel.call());
            }
        );
    };
};


//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[])
{
    int N;

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    // 1D vector-vector multiply
    N = 10000000;
    std::vector<double> x(N);
    std::iota(x.begin(), x.end(), 0.0);
    std::vector<double> y(N);
    std::iota(y.begin(), y.end(), 0.0);
    std::vector<double> z(N);
    std::vector<double> w(N);

    // use tag to label data params with unique IDs
    auto X = TAG(x);
    auto Y = TAG(y);
    auto Z = TAG(z);
    auto W = TAG(w);

    { // Some static assertions to make sure I understand the types
        static_assert(!std::is_const_v<EquivalentView<Kokkos::Serial, decltype(X)>::value_type>);
        static_assert(
          std::is_const_v<EquivalentView<Kokkos::Serial, decltype(std::as_const(X))>::value_type>);

        static_assert(
          std::is_same_v<decltype(Views<Kokkos::Serial>::create_view(std::as_const(X))),
                         EquivalentView<Kokkos::Serial, std::add_const_t<decltype(x)>>::type>);

        static_assert(std::is_same_v<decltype(Views<Kokkos::Serial>::create_view(X)),
                                     EquivalentView<Kokkos::Serial, decltype(x)>::type>);

        auto args = pack(std::as_const(X), Y);
        static_assert(
          std::is_same_v<decltype(args), std::tuple<decltype(std::as_const(X))&, decltype(Y)&>>);

        auto views = Views<Kokkos::Serial>::create_views_from_tuple(args);
        static_assert(std::is_same_v<
                      decltype(views),
                      std::tuple<EquivalentView<Kokkos::Serial, decltype(std::as_const(X))>::type,
                                 EquivalentView<Kokkos::Serial, decltype(Y)>::type>>);
    }

    Kernel k1(
      "1D vector-vector multiply",
      pack(std::as_const(X), std::as_const(Y), Z),
      []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
      {
          auto& x = std::get<0>(views);
          auto& y = std::get<1>(views);
          auto& z = std::get<2>(views);
          z[i]    = x[i] * y[i];
      },
      range_extent(0, z.size()));

    Kernel k2(
      "1D vector-vector multiply",
      pack(std::as_const(X), std::as_const(Z), W),
      []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
      {
          auto& x = std::get<0>(views);
          auto& z = std::get<1>(views);
          auto& w = std::get<2>(views);
          w[i]    = x[i] * z[i];
      },
      range_extent(0, w.size()));

    ////auto kernels = create_tuple_of_references(k1, k2);
    // auto kernels = pack(k1, k2);
    // auto kernel_graph = create_kernel_info_tuple(kernels);


    // Create an Algorithm object
    Algorithm algo(pack(k1, k2));

    // k1.data_params[0] should match k2.data_params[0]
    static_assert(DoTagsMatch<decltype(k1.data_params_), 0, decltype(k2.data_params_), 0>);

    // k1.data_params[0] should not match k2.data_params[1]
    static_assert(!DoTagsMatch<decltype(k1.data_params_), 0, decltype(k2.data_params_), 1>);



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
            []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i, const Index&
    j)
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
        using MatrixType = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

        MatrixType x(N, N); // 32x32
        x.setRandom();
        N -= 2;
        MatrixType y(N, N); // 30x30
        y.setZero();

        Kernel k(
          "2D convolution",
          pack(std::as_const(x), y),
          []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i, const Index& j)
          {
              auto& x_view = std::get<0>(views);
              auto x_subview =
                Kokkos::subview(x_view, Kokkos::make_pair(i, i + 3), Kokkos::make_pair(j, j + 3));
              auto& y_view = std::get<1>(views);

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
    // - perhaps use counter for each view (+1 for algorithm output) to know when to deallocate it
    // Need algorithm to construct the counters, for example:
    //   k.parameters[1] is not const and hence output
    //   k2.parameters[0] is const and hence input
    // assert(&std::get<1>(k.parameters) == &std::get<0>(k2.parameters));


    // execute all kernels
    ////algo._deduce_dependencies();
    // algo.call();

    // 1D vector-vector multiply: verify the output
    // for (auto i = 0; i < y.size(); i++) {
    //    assert(x[i] * y[i] == z[i]);
    //}

    // Finalize Kokkos
    Kokkos::finalize();

    printf("\n** GRACEFUL EXIT **\n");
}
