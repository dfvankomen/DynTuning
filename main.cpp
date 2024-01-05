// Assumptions:
// * Algorithms are cast as operating on one element at a time
// * Kernels are steps in the algorithm that typically involve a loop over the elements
// * Inputs and outputs are all ndarrays at the element level
//

#include <Kokkos_Core.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>

class Algorithm
{
};


// Used to map a container to the corresponding view type
template<typename ExecutionSpace, typename T>
struct EquivalentView;

template<typename ExecutionSpace, typename T>
struct EquivalentView<ExecutionSpace, std::vector<T>>
{
    using type = Kokkos::View<typename std::vector<T>::value_type*,
                              typename ExecutionSpace::array_layout,
                              typename ExecutionSpace::memory_space>;
};

template<typename ExecutionSpace, typename T>
struct EquivalentView<ExecutionSpace, const std::vector<T>>
{
    using type = Kokkos::View<const typename std::vector<T>::value_type*,
                              typename ExecutionSpace::array_layout,
                              typename ExecutionSpace::memory_space>;
};

template<typename ExecutionSpace>
struct Views
{
    // Create a view for a given executation space and C++ data structure(each structure needs a
    // specialization)
    template<typename T>
    static typename EquivalentView<ExecutionSpace, T>::type create_view(T&& arr)
    {
        // static_assert(false, "Specialization for type not implemented yet.");
    }

    // Specialization for std::vector (default allocator)
    template<typename T>
    static typename EquivalentView<ExecutionSpace, std::vector<T>>::type create_view(
      std::vector<T>& arr)
    {
        return
          typename EquivalentView<ExecutionSpace, std::vector<T>>::type(arr.data(), arr.size());
    }

    // Specialization for const std::vector (default allocator)
    template<typename T>
    static typename EquivalentView<ExecutionSpace, const std::vector<T>>::type create_view(
      const std::vector<T>& arr)
    {
        return typename EquivalentView<ExecutionSpace, const std::vector<T>>::type(arr.data(),
                                                                                   arr.size());
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



    /*
    template<typename... ParameterTypes>
    static auto create_views2(ParameterTypes&&... params)
    {
        return std::make_tuple(Views<ExecutionSpace>::create_view(params)...);
    }
    */
};

// Need method to get view from data type

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

template<typename LambdaType, typename... ParameterTypes>
class Kernel
{
  public:
    // Note: we are choosing the host and device excution space at compile time
    using HostExecutionSpace   = Kokkos::Serial;
    using DeviceExecutionSpace = Kokkos::Serial;

    Kernel(std::tuple<ParameterTypes&...> params, const LambdaType& lambda)
      : parameters(params)
      , kernel_lambda(lambda)
      , parameter_views_host(Views<HostExecutionSpace>::create_views_from_tuple(params))
      , parameter_views_device(Views<DeviceExecutionSpace>::create_views_from_tuple(parameters))
    {
    }

    void call()
    {

        // data movement needs to happen at the algorithm level
        //
        // Inside Kernel for the wrapper
        // TODO deep copies (somewhere)
        auto kernel_wrapper = [=](const auto& i)
        {
            kernel_lambda(parameter_views_host, i);
        };

        // Note: TODO think about how to make range policy generic to work for:
        // 1) Matrix-vector operation (different range)
        // 2) Matrix-matrix operation
        // 3) Vector-vector operation of same size
        // 4) vector-vector opertion with difference sizes (convolution)
        auto range_policy = Kokkos::RangePolicy<HostExecutionSpace>(HostExecutionSpace(),
                                                                    0,
                                                                    std::get<0>(parameters).size());

        Kokkos::parallel_for(range_policy, kernel_wrapper);
    };

    // data_host_references
    // data_views
    // execution_parameters = # of threads, etc.

    std::tuple<ParameterTypes&...> parameters;
    std::tuple<typename EquivalentView<HostExecutionSpace, ParameterTypes>::type...>
      parameter_views_host;
    std::tuple<typename EquivalentView<DeviceExecutionSpace, ParameterTypes>::type...>
      parameter_views_device;
    LambdaType kernel_lambda;
};

int main()
{
    std::vector<double> x(1000);
    std::iota(x.begin(), x.end(), 0.0);

    std::vector<double> y(1000);

    Kokkos::View<std::vector<double>> a;

    // auto views = Views<Kokkos::Serial>::create_views(x, y);

    // Kernels never allocate data (global)

    Kernel k(pack(std::as_const(x), y),
             []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
             {
                 auto& x = std::get<0>(views);
                 auto& y = std::get<1>(views);
                 y[i]    = x[i] * x[i];
             });

    k.call();

    for (auto i = 0; i < 1000; i++)
    {
        assert(x[i] * x[i] == y[i]);
    }

    std::vector<double> z(1000);

    Kernel k2(pack(std::as_const(y), z),
              []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
              {
                  auto& x = std::get<0>(views);
                  auto& y = std::get<1>(views);
                  y[i]    = x[i] * x[i];
              });

    // At the end, the algorithm needs to know the "final" output that needs copied to the host
    // Data needs moved if 1) it is a kernel input or 2) algorithm output
    // Data view deallocation if 1) it is not a downstream input 2) and not algorithm output
    // - perhaps use counter for each view (+1 for algorithm output) to know when to deallocate it
    // Need algorithm to construct the counters, for example:
    //   k.parameters[1] is not const and hence output
    //   k2.parameters[0] is const and hence input
    // assert(&std::get<1>(k.parameters) == &std::get<0>(k2.parameters));
}