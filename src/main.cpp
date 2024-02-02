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
//#include "Eigen"

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

class Algorithm
{
  // start with an array of kernels
  // deduce data dependecies among kernels
  // organize kernels into independent chains
  // store array of arrays of kernel pointers
  //   each subchain can be executed in any order
  //   kernels within each subchain must be executed in order
  // iterate over the kernels in subchains and determine for every output,
  //   find the next kernel that depends on that output,
  //   determine which device that kernel will run on,
  //   then immediately start "copying" the output to that device
  //   (if the souce and destination devices are the same it will be a no-op)

  // enum {INPUT, OUTPUT};
  // for each kernel
  //   for each arg in the parameters tuple
  //     if arg is const      // input
  //       flag = 0
  //     if arg is not const  // output
  //       flag = 1
  //    std::is_const<decltype(i)>::value
  //
  // for each output, does it appear as input for any kernels? if yes, count/track that
  // make sure all inputs are accounted for,
  //   they should be already present at the start of the algorithm,
  //   or
  //   they should be an output of a kernel in the algorithm
  // detect circular dependency and throw an error

  // kernel subchain order can be an additional permutative variable

  // read in a configuration file with bounds on the possible variables to optimize
  // generate all possible combinations of those variables
  // for each possible combination, call the algorithm and store timings

  // example
  /*

  Kernel k1(pack(std::as_const(a), b), lambda1);
  Kernel k2(pack(std::as_const(c), d), lambda2);
  Kernel k3(pack(std::as_const(b), e), lambda3);

  inputs: a, c, b
  outputs: b, d, e
  outinputs: b

  chains:
    chain1: k1, k3
    chain2: k2

  permuations of chains: (0,1) and (1,0)

  for each run:
    // for each output construct a list (in order) of devices for dependent kernels
    // how to mark whether an output should be copied back the host?
    data_dependencies:
      b: (device1)

  for each kernel after it runs, lookup the next device in the data_dependencies
    start a data copy to that device
    (if the source and destination are the same device, it will be a no-op)

  */

  // what if a kernel depends on more than one kernel, but suppose those kernels are independent of each other
  // they should be allowed to execute in any order
  // might be important if one takes longer to transfer data than the other
  // might be optimal to run one first then run the next while the data has started copying

};

// Concepts that will be used for EquivalentView

template<class, template<class...> class>
inline constexpr bool is_specialization = false;
template<template<class...> class T, class... Args>
inline constexpr bool is_specialization<T<Args...>, T> = true;

template<typename T>
concept IsStdVector = is_specialization<std::decay_t<T>, std::vector>;

//template<typename T>
//concept IsEigenMatrix = std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T> >, std::decay_t<T> >;

template<typename EigenT, typename KokkosLayout>
concept IsLayoutSame =
  (std::is_same_v<KokkosLayout, Kokkos::LayoutRight> && std::decay_t<EigenT>::IsRowMajor == 1) ||
  (std::is_same_v<KokkosLayout, Kokkos::LayoutLeft> && std::decay_t<EigenT>::IsRowMajor == 0);

template<typename T>
concept IsConst = std::is_const_v<std::remove_reference_t<T> >;

template<typename T>
concept NotConst = not std::is_const_v<std::remove_reference_t<T> >;


// Used to map a container to the corresponding view type
template<typename ExecutionSpace, typename T>
struct EquivalentView;

template<typename ExecutionSpace, typename T>
    requires IsStdVector<T>
    //requires IsStdVector<T> || IsEigenMatrix<T>

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
    //template<typename T>
    //    requires IsEigenMatrix<T>
    //static auto create_view(T& matrix)
    //{
    //    using ViewType = typename EquivalentView<ExecutionSpace, T>::type;
    //    return ViewType(matrix.data(), matrix.rows(), matrix.cols());
    //}

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
    using type = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<KernelRank> >;
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

/// @brief Packs a list of references to data containers into a tuple
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
        
        //debugging diagnostics
        printf("\nHost Execution Space:\n");
        HostExecutionSpace{}.print_configuration(std::cout);
        printf("\nDevice Execution Space:\n");
        DeviceExecutionSpace{}.print_configuration(std::cout);
      
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
        return (char*) kernel_name_.c_str();
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


#define TIMING(f) \
{ \
    Kokkos::Timer timer; \
    timer.reset(); \
    f; \
    double kernel_time = timer.seconds(); \
    printf("%s: %.6f\n", k.name(), kernel_time); \
}

int main(int argc, char* argv[])
{

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);

    // 1D vector-vector multiply
    {
        // set up data
        int N = 10000000;
        std::vector<double> x(N);
        std::iota(x.begin(), x.end(), 0.0);
        std::vector<double> y(N);
        std::iota(y.begin(), y.end(), 0.0);
        std::vector<double> z(N);

        // define the kernel
        Kernel k(
            "1D vector-vector multiply",
            pack(std::as_const(x), std::as_const(y), z),
            []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
            {
                auto& x = std::get<0>(views);
                auto& y = std::get<1>(views);
                auto& z = std::get<2>(views);
                z[i] = x[i] * y[i];
            },
            range_extent(0, z.size())
        );

        // run the kernel
        TIMING(k.call());
        
        // verify the output
        for (auto i = 0; i < y.size(); i++) {
            assert(x[i] * y[i] == z[i]);
        }
    }

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
            []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i, const Index& j)
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

  //Finalize Kokkos
  Kokkos::finalize();

  printf("\n** GRACEFUL EXIT **\n");
}
