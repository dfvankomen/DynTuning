// Assumptions:
// * Algorithms are cast as operating on one element at a time
// * Kernels are steps in the algorithm that typically involve a loop over the elements
// * Inputs and outputs are all ndarrays at the element level
//

// tmp
#include "Eigen"
#include "Kokkos_Core.hpp"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

class Algorithm
{
};

// Concepts that will be used for EquivalentView

template<class, template<class...> class>
inline constexpr bool is_specialization = false;
template<template<class...> class T, class... Args>
inline constexpr bool is_specialization<T<Args...>, T> = true;

template<typename T>
concept IsStdVector = is_specialization<std::decay_t<T>, std::vector>;

template<typename T>
concept IsEigenMatrix = std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>>;

template<typename T>
concept IsConst = std::is_const_v<std::remove_reference_t<T>>;

template<typename T>
concept NotConst = not std::is_const_v<std::remove_reference_t<T>>;


// Used to map a container to the corresponding view type
template<typename ExecutionSpace, typename T>
struct EquivalentView;

template<typename ExecutionSpace, typename T>
    requires IsStdVector<T> || IsEigenMatrix<T>
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

    // Specialization for Eigen matrix
    template<typename T>
        requires IsEigenMatrix<T>
    static auto create_view(T& matrix)
    {
        using ViewType = typename EquivalentView<ExecutionSpace, T>::type;
        return ViewType(matrix.data(), matrix.rows(), matrix.cols());
    }


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
    using HostExecutionSpace   = Kokkos::Serial;
    using DeviceExecutionSpace = Kokkos::Serial;
    using BoundType            = RangeExtent<KernelRank>::value_type;

    Kernel(std::tuple<ParameterTypes&...> params,
           const LambdaType& lambda,
           const RangeExtent<KernelRank>& range_extent)
      : data_params_(params)
      , kernel_lambda_(lambda)
      , data_views_host_(Views<HostExecutionSpace>::create_views_from_tuple(params))
      , data_views_device_(Views<DeviceExecutionSpace>::create_views_from_tuple(params))
      , lower_(range_extent.lower)
      , upper_(range_extent.upper)
    {
    }

    virtual void call()
    {
        // data movement needs to happen at the algorithm level
        //
        // Inside Kernel for the wrapper
        // TODO deep copies (somewhere)

        auto kernel_wrapper = [=](const auto... indices)
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

int main()
{
    // 1D vector-vector multiply
    {
        int N = 1000;

        std::vector<double> x(N);
        std::iota(x.begin(), x.end(), 0.0);

        std::vector<double> y(N);

        auto e = range_extent(0, y.size());

        Kernel k(
          pack(std::as_const(x), y),
          []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
          {
              auto& x = std::get<0>(views);
              auto& y = std::get<1>(views);
              y[i]    = x[i] * x[i];
          },
          range_extent(0, y.size()));

        using ExtentType1 = RangeExtent<1>;

        k.call();

        for (auto i = 0; i < y.size(); i++)
        {
            assert(x[i] * x[i] == y[i]);
        }
    }

    // 2D convolution
    {
        unsigned int N = 32;

        Eigen::MatrixXd x(N, N); // 32x32
        x.setRandom();

        N -= 2;
        Eigen::MatrixXd y(N, N); // 30x30

        auto x_view = Views<Kokkos::Serial>::create_view(x);
        static_assert(x_view.Rank == 2, "We need to figure out why the view is Rank 1...");

        // Uncomment after the above issue is resolved
        /*
        Kernel k(
          pack(std::as_const(x), y),
          []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i, const Index& j)
          {
              auto& x_view = std::get<0>(views); // 3x3 subview
              auto x =
                Kokkos::subview(x_view, Kokkos::make_pair(i, j), Kokkos::make_pair(i + 3, j + 3));
              auto& y = std::get<1>(views);
              y[i, j] = 0;
              for (int ii = 0; ii < 3; ii++)
                  for (int jj = 0; jj < 3; jj++)
                      y[i, j] += x[ii, jj];
          },
          range_extent({ 0, 0 }, { N, N }));

        k.call();

        for (auto i = 0; i < N; i++)
        {
            for (auto j = 0; j < N; j++)
            {
                auto tmp = 0.0;
                for (auto ii = 0; ii < 3; ii++)
                {
                    for (auto jj = 0; jj < 3; jj++)
                    {
                        auto iii = i + ii;
                        auto jjj = j + jj;
                        tmp += x(iii, jjj);
                    }
                }
                assert(tmp == y(i, j));
            }
        }
        */
    }
    /*
    // 3D convolution
    {
        int N = 10;

        std::vector<double> x(N * N * N); // 10x10x10
        std::iota(x.begin(), x.end(), 0.0);
        Kokkos::mdspan x_span { x.data(), N, N, N };

        N -= 2;
        std::vector<double> y(N * N * N); // 8x8x8
        Kokkos::mdspan y_span { y.data(), N, N, N };

        Kernel3D k(pack(std::as_const(x_span), y_span),
                   []<typename ViewsTuple, typename Index>(ViewsTuple& views,
                                                           const Index& i,
                                                           const Index& j,
                                                           const Index& k)
                   {
                       auto& x    = std::get<0>(views)(i + 3, j + 3, k + 3); // 3x3x3 subview
                       auto& y    = std::get<1>(views);
                       y[i, j, k] = 0;
                       for (int ii = 0; ii < 3; ii++)
                           for (int jj = 0; jj < 3; jj++)
                               for (int kk = 0; kk < 3; kk++)
                                   y[i, j, k] += x[ii, jj, kk];
                   },
                   { 0, 0, 0 },
                   { y_span.extent(0), y_span.extent(1), y_span.extent(2) });

        k.call();

        for (auto i = 0; i < y_span.extent(0); i++)
        {
            for (auto j = 0; j < y_span.extent(1); j++)
            {
                for (auto k = 0; k < y_span.extent(2); k++)
                {
                    auto tmp = 0.0;
                    for (auto ii = 0; ii < 3; ii++)
                    {
                        for (auto jj = 0; jj < 3; jj++)
                        {
                            for (auto kk = 0; kk < 3; kk++)
                            {
                                auto iii = i + ii;
                                auto jjj = j + jj;
                                auto kkk = k + kk;
                                tmp += x_span(iii, jjj, kkk);
                            }
                        }
                    }
                    assert(tmp == y_span(i, j, k));
                }
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
}
