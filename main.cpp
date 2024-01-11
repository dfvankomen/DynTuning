// Assumptions:
// * Algorithms are cast as operating on one element at a time
// * Kernels are steps in the algorithm that typically involve a loop over the elements
// * Inputs and outputs are all ndarrays at the element level
//

//tmp
#include <Kokkos_Core.hpp>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <vector>
#include <mdspan>

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
    typedef Kokkos::Serial HostExecutionSpace;
    typedef Kokkos::Serial DeviceExecutionSpace;

    Kernel(std::tuple<ParameterTypes&...> params, const LambdaType& lambda, point_type lower, point_type upper)
      : parameters(params)
      , kernel_lambda(lambda)
      , parameter_views_host(Views<HostExecutionSpace>::create_views_from_tuple(params))
      , parameter_views_device(Views<DeviceExecutionSpace>::create_views_from_tuple(params))
      , m_lower(lower)
      , m_upper(upper)
    {
    }

    virtual void call()
    {
        // data movement needs to happen at the algorithm level
        //
        // Inside Kernel for the wrapper
        // TODO deep copies (somewhere)

        // make kernel_wrapper variadic to handle multi-dimensional indexing
       
        // Note: TODO think about how to make range policy generic to work for:
        // 1) Matrix-vector operation (different range)
        // 2) Matrix-matrix operation
        // 3) Vector-vector operation of same size
        // 4) vector-vector opertion with difference sizes (convolution)
        
        // range_policy and kernel_wrapper need to match in dimensionality
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

template<typename LambdaType, typename... ParameterTypes>
class Kernel1D : public Kernel<LambdaType, ParameterTypes...>
{
  public:
    
    typedef Kernel<LambdaType, ParameterTypes...>::HostExecutionSpace HostExecutionSpace;
    typedef Kernel<LambdaType, ParameterTypes...>::DeviceExecutionSpace DeviceExecutionSpace;
    
    using array_index_type = std::int64_t;
    using kernel_rank = Kokkos::Rank<1>;
    using point_type = Kokkos::Array<array_index_type, kernel_rank>;
    using tile_type  = Kokkos::Array<array_index_type, kernel_rank>;

    Kernel2D(std::tuple<ParameterTypes&...> params, const LambdaType& lambda, point_type lower, point_type upper)
      : Kernel<LambdaType, ParameterTypes...>(params, lambda)
      , m_lower(lower)
      , m_upper(upper)
    {
    };

    void call()
    {
        auto kernel_wrapper = [=](const auto... indices)
        {
            kernel_lambda(parameter_views_host, indices...);
        };
       
        auto range_policy = Kokkos::MDRangePolicy<HostExecutionSpace>(HostExecutionSpace,
                                                                      kernel_rank,
                                                                      m_lower,
                                                                      m_upper);

        Kokkos::parallel_for(range_policy, kernel_wrapper);
    };
    
    point_type m_lower;
    point_type m_upper;
    tile_type m_tile;
};

template<typename LambdaType, typename... ParameterTypes>
class Kernel2D : public Kernel<LambdaType, ParameterTypes...>
{
  public:
    
    typedef Kernel<LambdaType, ParameterTypes...>::HostExecutionSpace HostExecutionSpace;
    typedef Kernel<LambdaType, ParameterTypes...>::DeviceExecutionSpace DeviceExecutionSpace;
    
    using array_index_type = std::int64_t;
    using kernel_rank = Kokkos::Rank<2>;
    using point_type = Kokkos::Array<array_index_type, kernel_rank>;
    using tile_type  = Kokkos::Array<array_index_type, kernel_rank>;

    Kernel2D(std::tuple<ParameterTypes&...> params, const LambdaType& lambda, point_type lower, point_type upper)
      : Kernel<LambdaType, ParameterTypes...>(params, lambda)
      , m_lower(lower)
      , m_upper(upper)
    {
    };

    void call()
    {
        auto kernel_wrapper = [=](const auto... indices)
        {
            kernel_lambda(parameter_views_host, indices...);
        };
       
        auto range_policy = Kokkos::MDRangePolicy<HostExecutionSpace>(HostExecutionSpace,
                                                                      kernel_rank,
                                                                      m_lower,
                                                                      m_upper);

        Kokkos::parallel_for(range_policy, kernel_wrapper);
    };
    
    point_type m_lower;
    point_type m_upper;
    tile_type m_tile;
};

template<typename LambdaType, typename... ParameterTypes>
class Kernel3D : public Kernel<LambdaType, ParameterTypes...>
{
  public:
    
    typedef Kernel<LambdaType, ParameterTypes...>::HostExecutionSpace HostExecutionSpace;
    typedef Kernel<LambdaType, ParameterTypes...>::DeviceExecutionSpace DeviceExecutionSpace;
    
    using array_index_type = std::int64_t;
    using kernel_rank = Kokkos::Rank<3>;
    using point_type = Kokkos::Array<array_index_type, kernel_rank>;
    using tile_type  = Kokkos::Array<array_index_type, kernel_rank>;

    Kernel2D(std::tuple<ParameterTypes&...> params, const LambdaType& lambda, point_type lower, point_type upper)
      : Kernel<LambdaType, ParameterTypes...>(params, lambda)
      , m_lower(lower)
      , m_upper(upper)
    {
    };

    void call()
    {
        auto kernel_wrapper = [=](const auto... indices)
        {
            kernel_lambda(parameter_views_host, indices...);
        };
       
        auto range_policy = Kokkos::MDRangePolicy<HostExecutionSpace>(HostExecutionSpace,
                                                                      kernel_rank,
                                                                      m_lower,
                                                                      m_upper);

        Kokkos::parallel_for(range_policy, kernel_wrapper);
    };
    
    point_type m_lower;
    point_type m_upper;
    tile_type m_tile;
};

int main()
{
    // 1D vector-vector multiply
    {
      int N = 1000;

      std::vector<double> x(N);
      std::iota(x.begin(), x.end(), 0.0);

      std::vector<double> y(N);
    
      Kernel1D k(pack(std::as_const(x), y),
               []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
               {
                   auto& x = std::get<0>(views);
                   auto& y = std::get<1>(views);
                   y[i]    = x[i] * x[i];
               },
               {0},
               {y.size()}
               );

      k.call();

      for (auto i = 0; i < y.size(); i++)
      {
          assert(x[i] * x[i] == y[i]);
      }
    }

    // 2D convolution
    {
      int N = 32;
      
      std::vector<double> x(N * N); // 32x32
      std::iota(x.begin(), x.end(), 0.0);
      std::mdspan x_span{x.data(), N, N};

      N -= 2;
      std::vector<double> y(N * N); //30x30
      std::mdspan y_span{y.data(), N, N};
    
      Kernel2D k(pack(std::as_const(x_span), y_span),
               []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i, const Index& j)
               {
                   auto& x = std::get<0>(views)(i:(i+3), j:(j+3)); // 3x3 subview
                   auto& y = std::get<1>(views);
                   y[i, j] = 0;
                   for (int ii = 0; ii < 3; ii++)
                     for (int jj = 0; jj < 3; jj++)
                       y[i, j] += x[ii, jj];
               }
               {0,0},
               {y.extent(0), y.extent(1)}
               );

      k.call();

      for (auto i = 0; i < y.extent(0); i++) {
        for (auto j = 0; j < y.extent(1); j++ {
          tmp = 0.0;
          for (auto ii = 0; ii < 3; ii++) {
            for (auto jj = 0; jj < 3; jj++) {
              auto iii = i + ii;
              auto jjj = j + jj;
              tmp += x[iii, jjj];
            }
          }
          assert(tmp == y[i,j]);
        }
      }
    }

    // 3D convolution
    {
      int N = 10;
      
      std::vector<double> x(N * N * N); // 10x10x10
      std::iota(x.begin(), x.end(), 0.0);
      std::mdspan x_span{x.data(), N, N, N};

      N -= 2;
      std::vector<double> y(N * N * N); //8x8x8
      std::mdspan y_span{y.data(), N, N, N};
    
      Kernel3D k(pack(std::as_const(x_span), y_span),
               []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i, const Index& j, const Index& k)
               {
                   auto& x = std::get<0>(views)(i:(i+3), j:(j+3), k:k(k+3)); // 3x3x3 subview
                   auto& y = std::get<1>(views);
                   y[i, j, k] = 0;
                   for (int ii = 0; ii < 3; ii++)
                     for (int jj = 0; jj < 3; jj++)
                       for (int kk = 0; kk < 3; kk++)
                         y[i, j, k] += x[ii, jj, kk];
               }
               {0,0,0},
               {y.extent(0), y.extent(1), y.extent(2)}
               );

      k.call();

      for (auto i = 0; i < y.extent(0); i++) {
        for (auto j = 0; j < y.extent(1); j++ {
          for (auto k = 0; k < y.extent(2); k++ {
            tmp = 0.0;
            for (auto ii = 0; ii < 3; ii++) {
              for (auto jj = 0; jj < 3; jj++) {
                for (auto kk = 0; kk < 3; kk++) {
                  auto iii = i + ii;
                  auto jjj = j + jj;
                  auto kkk = k + kk;
                  tmp += x[iii, jjj, kkk];
                }
              }
            }
            assert(tmp == y[i,j,k]);
          }
        }
      }
    }

    //Kokkos::View<std::vector<double>> a;

    // auto views = Views<Kokkos::Serial>::create_views(x, y);

    // Kernels never allocate data (global)

    /*
    std::vector<double> z(1000);

    Kernel k2(pack(std::as_const(y), z),
              []<typename ViewsTuple, typename Index>(ViewsTuple& views, const Index& i)
              {
                  auto& x = std::get<0>(views);
                  auto& y = std::get<1>(views);
                  y[i]    = x[i] * x[i];
              });
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
