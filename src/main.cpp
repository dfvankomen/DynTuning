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
// #include "Eigen"

#define NDEBUG
#ifdef NDEBUG
#include <iostream>
#endif


//



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


// Concepts that will be used for EquivalentView

template<class, template<class...> class>
inline constexpr bool is_specialization = false;
template<template<class...> class T, class... Args>
inline constexpr bool is_specialization<T<Args...>, T> = true;

template<typename T>
concept IsStdVector = is_specialization<std::decay_t<T>, std::vector>;

// template<typename T>
// concept IsEigenMatrix = std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T> >, std::decay_t<T> >;

template<typename EigenT, typename KokkosLayout>
concept IsLayoutSame =
  (std::is_same_v<KokkosLayout, Kokkos::LayoutRight> && std::decay_t<EigenT>::IsRowMajor == 1) ||
  (std::is_same_v<KokkosLayout, Kokkos::LayoutLeft> && std::decay_t<EigenT>::IsRowMajor == 0);

template<typename T>
concept IsConst = std::is_const_v<std::remove_reference_t<T>>;

template<typename T>
concept NotConst = not std::is_const_v<std::remove_reference_t<T>>;


// Used to map a container to the corresponding view type
template<typename ExecutionSpace, typename T>
struct EquivalentView;

template<typename ExecutionSpace, typename T>
    requires IsStdVector<T>
struct EquivalentView<ExecutionSpace, T>
{
    // Type of the scalar in the data structure
    using value_type = std::conditional_t<std::is_const_v<T>,
                                          const typename std::remove_reference_t<T>::value_type,
                                          typename std::remove_reference_t<T>::value_type>;

    // Type for the equivalent view of the data structure
    using type = Kokkos::View<value_type*,
                              typename ExecutionSpace::array_layout,
                              typename ExecutionSpace::memory_space>;
};

// template<typename ExecutionSpace, typename T>
//     requires IsEigenMatrix<T>
// struct EquivalentView<ExecutionSpace, T>
//{
//     // Type of the scalar in the data structure
//     using value_type = std::conditional_t<std::is_const_v<T>,
//                                           const typename std::remove_reference_t<T>::value_type,
//                                           typename std::remove_reference_t<T>::value_type>;
//
//     // Type for the equivalent view of the data structure
//     using type = Kokkos::View<value_type**,
//                               typename ExecutionSpace::array_layout,
//                               typename ExecutionSpace::memory_space>;
// };

template<typename ExecutionSpace>
struct Views
{
    // Create a view for a given executation space and C++ data structure(each structure needs a
    // specialization)
    template<typename T>
    static typename EquivalentView<ExecutionSpace, T>::type create_view(T&&);

    // Specialization for std::vector (default allocator)
    template<typename T>
        requires IsStdVector<T>
    static auto create_view(T& vector)
    {
        using ViewType = typename EquivalentView<ExecutionSpace, T>::type;
        return ViewType(vector.data(), vector.size());
    }

    //// Specialization for Eigen matrix
    // template<typename T>
    //     requires IsEigenMatrix<T>
    // static auto create_view(T& matrix)
    //{
    //     using ViewType = typename EquivalentView<ExecutionSpace, T>::type;
    //     return ViewType(matrix.data(), matrix.rows(), matrix.cols());
    // }

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
    static auto create_views_from_tuple(std::tuple<ParameterTypes...>& params_tuple)
    {
        return create_views_helper(params_tuple,
                                   std::make_index_sequence<sizeof...(ParameterTypes)> {});
    }
};

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


RangeExtent<1> range_extent(const ArrayIndex& lower, const ArrayIndex& upper)
{
    return { lower, upper };
}

RangeExtent<2> range_extent(const Kokkos::Array<ArrayIndex, 2>& lower,
                            const Kokkos::Array<ArrayIndex, 2>& upper)
{
    return { lower, upper };
}

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
constexpr std::vector<std::size_t> match_input_data_param_helper(const KernelsTuple& algorithm_kernels,
                                             std::integer_sequence<std::size_t, L...>)
{
    auto& this_kernel     = std::get<I>(algorithm_kernels);
    auto& this_data       = std::get<J>(this_kernel.data_params_);
    auto& compared_kernel = std::get<K>(algorithm_kernels);

    return { [&](auto& compared_data) -> std::size_t
             {
                 if (&this_data == &compared_data && std::is_const_v<decltype(compared_data)>)
                 {
                     return L;
                 }
                 else
                 {
                     return std::numeric_limits<std::size_t>::max();
                 }
             }(std::get<L>(compared_kernel.data_params_))... };
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

    auto& this_kernel     = std::get<I>(algorithm_kernels);
    auto& this_data       = std::get<J>(this_kernel.data_params_);
    auto& compared_kernel = std::get<K>(algorithm_kernels);

    // Loop over the data parameters in kernel[K] and look for a match to kernel[I]->data[J]
    // that is also not const (an output)
    auto matches = match_input_data_param_helper<KernelsTuple, I, J, K>(
      algorithm_kernels,
      std::make_index_sequence<std::tuple_size_v<decltype(compared_kernel.data_params_)>> {});

    return *std::min_element(matches.begin(), matches.end());
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
    lambda(elem);
    iter_tuple<LambdaType, I + 1, T...>(t, lambda);
}

// utility function to check whether a parameter is const
template<typename ParamType>
bool is_const(ParamType& param)
{
    return false;
}
template<typename ParamType>
bool is_const(const ParamType& param)
{
    return true;
}

//// Helper function to create a tuple of references
// template <typename... Args>
// auto create_tuple_of_references(Args&&... args) {
//       return std::forward_as_tuple(std::forward<Args>(args)...);
// }
////auto tuple_of_references = create_tuple_of_references(k1, k2, k3);

// 1) start with some kernels
//  Kernel k1(...);
//  Kernel k2(...);
//  kernel k3(...);

// 2) create a tuple of KernelInfo, one for each kernel

// utility to create a tuple of N objects of same type T at compile time
template<size_t I, typename T>
struct tuple_n
{
    template<typename... Args>
    using type = typename tuple_n<I - 1, T>::template type<T, Args...>;
};

template<typename T>
struct tuple_n<0, T>
{
    template<typename... Args>
    using type = std::tuple<Args...>;
};
template<size_t I, typename T>
using tuple_of = typename tuple_n<I, T>::template type<>;

// helper function to create a tuple of T* based on the number data params a kernel has
template<typename T, typename KernelType>
constexpr std::tuple<T*> create_kernel_info_ptr_tuple(KernelType& k)
{
    return tuple_of<std::tuple_size<decltype(k.data_params_)>::value, T*> {};
}

// struct to act as nodes in the computation graph
template<typename KernelType>
struct KernelInfo
{
    constexpr KernelInfo(KernelType& k)
      : kernel(k)
      , next(create_kernel_info_ptr_tuple<KernelInfo>(k)) {};
    KernelType& kernel;
    std::tuple<KernelInfo*> next;
};

// Helper function to initialize an instance of KernelInfo
template<typename KernelType>
constexpr KernelInfo<KernelType> create_kernel_info(KernelType& k)
{
    return { k };
}

// Helper function to initialize a tuple of KernelInfo from a pack of Kernels
template<typename KernelType, size_t... Is>
constexpr auto create_kernel_info_tuple_impl(KernelType& k, std::index_sequence<Is...>)
{
    // return std::make_tuple(create_kernel_info<KernelType>(std::get<Is>(k))...);
    return std::make_tuple(
      create_kernel_info<std::tuple_element_t<Is, KernelType>>(std::get<Is>(k))...);
}
template<typename KernelType>
constexpr auto create_kernel_info_tuple(KernelType& k)
{
    return create_kernel_info_tuple_impl(
      k,
      std::make_index_sequence<std::tuple_size_v<KernelType>> {});
}
*/

// main algorithm function
template<typename... KernelTypes>
class Algorithm
{
  public:
    // constructor should initialize and empty vector
    constexpr Algorithm(std::tuple<KernelTypes&...> kernels)
      : kernels_(kernels)
      //, kernel_graph_(create_kernel_info_tuple(kernels))
      {
          // iter_tuple(kernels_,
          //            []<typename KernelType>(KernelType& kernel)
          //            { printf("Registered Kernel: %s\n", kernel.kernel_name_.c_str()); });
      };
    ~Algorithm() {};

    // the core of this class is a tuple of kernels
    std::tuple<KernelTypes&...> kernels_;
    // std::tuple<KernelInfo<KernelTypes>...> kernel_graph_;

    /*
    // deduce data relationships
    void _deduce_dependencies()
    {
      iter_tuple(kernels_, []<typename KernelType>(KernelType& kernel) {
        printf("Kernel: %s\n", kernel.kernel_name_.c_str());
        iter_tuple(kernel.data_params_, []<typename ParamType>(ParamType& param) {
          // NOTE can access name of a view with view.label()
          //   if param has no dependents set next_device == HOST
          //   if param has at least 1 dependent
          //     set next_device to the device of the next kernel that depends on it
          //     each kernel should have a next_device for each param
          //     execute a copy for each param to their own next_device
          if (check_is_const(param))
            printf("param is const\n");
          else
            printf("param is not const\n");
        });
      });
    };
    */

    // call all kernels
    void call()
    {
        iter_tuple(kernels_,
                   []<typename KernelType>(KernelType& kernel) { TIMING(kernel, kernel.call()); });
    };
};


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

    Kernel k1(
      "1D vector-vector multiply",
      pack(std::as_const(x), std::as_const(y), z),
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
      pack(std::as_const(x), std::as_const(z), w),
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

    constexpr auto match = match_input_data_param<0, 0, 0>(algo.kernels_);



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
