#pragma once
#include "Kokkos_Core.hpp"
#include "common.hpp"

#include <type_traits>

using HostExecutionSpace   = Kokkos::KOKKOS_HOST;
using DeviceExecutionSpace = Kokkos::KOKKOS_DEVICE;

// view memory type options
enum class ViewMemoryType
{
    NONOWNING,
    TMP,
    OWNING
};

// space
using HostViewSpace    = HostExecutionSpace;
using DeviceViewSpace  = DeviceExecutionSpace;
using ScratchViewSpace = HostViewSpace;

// layout
using HostViewLayout    = typename HostExecutionSpace::array_layout;
using DeviceViewLayout  = Kokkos::LayoutLeft;
using ScratchViewLayout = DeviceViewLayout;

// generator
template<typename ExecutionSpace,
         ViewMemoryType MemoryType,
         typename ArrayLayoutType = typename ExecutionSpace::array_layout>
struct Views;
using HostViewGenerator    = Views<HostViewSpace, ViewMemoryType::NONOWNING>;
using DeviceViewGenerator  = Views<DeviceViewSpace, ViewMemoryType::OWNING>;
using ScratchViewGenerator = Views<ScratchViewSpace, ViewMemoryType::TMP>;

// convenience defs for users
template<typename T>
using NDArrayView = Kokkos::View<T, Kokkos::HostSpace>;


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

template<typename T>
concept IsEigenMatrix = std::is_base_of_v<Eigen::MatrixBase<std::decay_t<T>>, std::decay_t<T>>;

template<typename T>
concept IsEigenTensor = requires {
    // NOTE: requires like this checks **all** blocks here to identify if it's true, these three are
    // a good way to check for Eigen tensors

    // Eigen Tensor, vs matrix, has a NumDimensions static member, so we'll check that first
    { std::decay_t<T>::NumDimensions } -> std::convertible_to<int>;

    // then we can check if it has a data pointer
    {
        std::declval<std::decay_t<T>>().data()
    } -> std::convertible_to<typename std::decay_t<T>::Scalar*>;

    // then finally see if there's a method for dimensions
    { std::declval<std::decay_t<T>>().dimensions() };
};

template<typename T>
concept IsKokkosView = is_specialization<std::decay_t<T>, Kokkos::View>;

//=============================================================================
// NRankViewType
//=============================================================================

// useful helper that allows for better template handling of the ranks
template<typename ScalarType, size_t Rank>
struct NRankViewType
{
    using type = ScalarType;
};

template<typename ScalarType>
struct NRankViewType<ScalarType, 1>
{
    using type = ScalarType*;
};
template<typename ScalarType>
struct NRankViewType<ScalarType, 2>
{
    using type = ScalarType**;
};
template<typename ScalarType>
struct NRankViewType<ScalarType, 3>
{
    using type = ScalarType***;
};
template<typename ScalarType>
struct NRankViewType<ScalarType, 4>
{
    using type = ScalarType****;
};
template<typename ScalarType>
struct NRankViewType<ScalarType, 5>
{
    using type = ScalarType*****;
};
template<typename ScalarType>
struct NRankViewType<ScalarType, 6>
{
    using type = ScalarType******;
};
template<typename ScalarType>
struct NRankViewType<ScalarType, 7>
{
    using type = ScalarType*******;
};
template<typename ScalarType>
struct NRankViewType<ScalarType, 8>
{
    using type = ScalarType********;
};

//=============================================================================
// EquivalentView
//=============================================================================

// https://kokkos.org/kokkos-core-wiki/API/core/KokkosConcepts.html
// https://kokkos.org/kokkos-core-wiki/API/core/view/view.html

/// Used to map a container to the corresponding view type
template<typename ExecutionSpace, typename MemoryLayout, typename T>
struct EquivalentView;

template<typename ExecutionSpace, typename MemoryLayout, typename T>
    requires IsStdVector<T>
struct EquivalentView<ExecutionSpace, MemoryLayout, T>
{

    // Type of the scalar in the data structure
    // Note: this ignores constness on purpose to allow deep-copies of "inputs" to kernels
    using value_type = typename std::remove_reference_t<T>::value_type;
    // using value_type = std::conditional_t<std::is_const_v<T>,
    //                    std::add_const_t<typename std::remove_reference_t<T>::value_type>,
    //                    typename std::remove_reference_t<T>::value_type>;

    // Type for the equivalent view of the data structure
    using type = Kokkos::View<value_type*, MemoryLayout, ExecutionSpace>;
    // using type = Kokkos::View<value_type*,
    //                           typename ExecutionSpace::array_layout,
    //                           typename ExecutionSpace::memory_space>;
};

template<typename ExecutionSpace, typename MemoryLayout, typename T>
    requires IsEigenMatrix<T>
struct EquivalentView<ExecutionSpace, MemoryLayout, T>
{
    // Type of the scalar in the data structure
    // Note: this ignores constness on purpose to allow deep-copies of "inputs" to kernels
    using value_type = typename std::remove_reference_t<T>::value_type;

    // Type for the equivalent view of the data structure
    using type = Kokkos::View<value_type**, MemoryLayout, ExecutionSpace>;
};

template<typename ExecutionSpace, typename MemoryLayout, typename T>
    requires IsEigenTensor<T>
struct EquivalentView<ExecutionSpace, MemoryLayout, T>
{
    // Type of the scalar in the data structure
    // Note: this ignores constness on purpose to allow deep-copies of "inputs" to kernels
    using value_type_scalar   = typename std::remove_reference_t<T>::Scalar;
    static constexpr int Rank = T::NumDimensions;
    using value_type          = typename NRankViewType<value_type_scalar, Rank>::type;

    // Type for the equivalent view of the data structure
    using type = Kokkos::View<value_type, MemoryLayout, ExecutionSpace>;
};

template<typename ExecutionSpace, typename MemoryLayout, typename T>
    requires IsKokkosView<T>
struct EquivalentView<ExecutionSpace, MemoryLayout, T>
{
    // Type of the scalar in the data structure
    // Note: this ignores constness on purpose to allow deep-copies of "inputs" to kernels

    // Kokkos's `data_type` includes the *'s that are necessary to determine size
    using value_type_scalar   = typename T::value_type;
    static constexpr int Rank = T::rank();
    using value_type          = typename NRankViewType<value_type_scalar, Rank>::type;

    // Type for the equivalent view of the data structure
    // Note that Kokkos views require the ** at the end, so we will just assume
    // that the view is set up properly by the user
    using type = Kokkos::View<value_type, MemoryLayout, ExecutionSpace>;
};

//=============================================================================
// Views
//=============================================================================

template<typename ExecutionSpace, ViewMemoryType MemoryType, typename ArrayLayoutType>
struct Views
{
    // Create a view for a given executation space and C++ data structure
    // (each structure needs a specialization)
    template<typename T>
    static typename EquivalentView<ExecutionSpace, Kokkos::LayoutLeft, T>::type create_view(T&);

    // Specialization for std::vector (default allocator)
    template<typename T>
        requires IsStdVector<T>
    static auto create_view(T& vector)
    {
        // Before you panic future developer...this const_cast is a neccessity to allow "inputs" to
        // be copied to the device.  However, we do need to find a way to restore constness later
        // for the sake of the kernel lambda code.

        // host layout same as device
        if constexpr (MemoryType == ViewMemoryType::NONOWNING)
        {
            using ViewType = typename EquivalentView<ExecutionSpace,
                                                     typename ExecutionSpace::array_layout,
                                                     T>::type;
            // using ViewType = typename EquivalentView<ExecutionSpace, T>::type;
            return ViewType(const_cast<ViewType::value_type*>(vector.data()), vector.size());
        }
        // vectors don't need tmp space
        else if constexpr (MemoryType == ViewMemoryType::TMP)
        {
            using ViewType = typename EquivalentView<ExecutionSpace, Kokkos::LayoutLeft, T>::type;
            return ViewType("", 0);
        }
        // device with ideal device layout
        else if constexpr (MemoryType == ViewMemoryType::OWNING)
        {
            using ViewType = typename EquivalentView<ExecutionSpace, Kokkos::LayoutLeft, T>::type;
            return ViewType("", vector.size());
        }
    }

    // Specialization for Eigen matrix
    template<typename T>
        requires IsEigenMatrix<T>
    static auto create_view(T& matrix)
    {
        // host layout may not be ideal for device
        if constexpr (MemoryType == ViewMemoryType::NONOWNING)
        {
            // NOTE: Eigen, by default uses "layoutleft" (meaning col-major order).
            // this differs from the default view which is LayoutRight.
            // originally the typename Kokkos::LayoutLeft was actually
            // typename ExecutationSpace::array_layout
            // TODO: a constexpr for if it's row major or not
            using ViewType = typename EquivalentView<ExecutionSpace,
                                                     typename ExecutionSpace::array_layout,
                                                     T>::type;
            //   typename EquivalentView<ExecutionSpace, typename Kokkos::LayoutLeft, T>::type;

            // using ViewType = typename EquivalentView<ExecutionSpace, T>::type;
            return ViewType(const_cast<ViewType::value_type*>(matrix.data()),
                            static_cast<size_t>(matrix.rows()),
                            static_cast<size_t>(matrix.cols()));
        }
        // tmp space on host with same layout as the device
        else if constexpr (MemoryType == ViewMemoryType::TMP)
        {
            // TODO: this is kind of dirty/hacky, as we should be passing the data in via
            // ArrayLayoutType instead of ScratchViewLayout, but this gets it working properly for
            // now
            using ViewType = typename EquivalentView<ExecutionSpace, ScratchViewLayout, T>::type;
            return ViewType("",
                            static_cast<size_t>(matrix.rows()),
                            static_cast<size_t>(matrix.cols()));
        }
        // device with ideal device layout
        else if constexpr (MemoryType == ViewMemoryType::OWNING)
        {
            // TODO: should this be using our template type?
            using ViewType = typename EquivalentView<ExecutionSpace,
                                                     typename ExecutionSpace::array_layout,
                                                     T>::type;
            return ViewType("", matrix.rows(), matrix.cols());
        }
    }

    // Specialization for Eigen Tensor (3d matrix, for example)
    template<typename T>
        requires IsEigenTensor<T>
    static auto create_view(T& tensor)
    {
        const auto& dimensions = tensor.dimensions();

        // this extents gives us our size_t from Eigen::array of dimensions
        auto extents_tuple = [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            return std::make_tuple(static_cast<size_t>(dimensions[Is])...);
        }(std::make_index_sequence<std::decay_t<T>::NumDimensions> {});

        // host layout may not be ideal for device
        if constexpr (MemoryType == ViewMemoryType::NONOWNING)
        {
            // NOTE: Eigen, by default uses "layoutleft" (meaning col-major order).
            // this differs from the default view which is LayoutRight.
            // originally the typename Kokkos::LayoutLeft was actually
            // typename ExecutationSpace::array_layout
            // TODO: a constexpr for if it's row major or not
            using ViewType = typename EquivalentView<ExecutionSpace,
                                                     typename ExecutionSpace::array_layout,
                                                     T>::type;

            return std::apply(
              [&](auto... extents)
            {
                return ViewType(const_cast<typename ViewType::value_type*>(tensor.data()),
                                extents...);
            },
              extents_tuple);

            // return ViewType(const_cast<ViewType::value_type*>(tensor.data()),
            //                 static_cast<size_t>(tensor.pages()),
            //                 static_cast<size_t>(tensor.rows()),
            //                 static_cast<size_t>(tensor.cols()));
        }
        // tmp space on host with same layout as the device
        else if constexpr (MemoryType == ViewMemoryType::TMP)
        {
            // TODO: this is kind of dirty/hacky, as we should be passing the data in via
            // ArrayLayoutType instead of ScratchViewLayout, but this gets it working properly for
            // now
            using ViewType = typename EquivalentView<ExecutionSpace, ScratchViewLayout, T>::type;

            return std::apply([&](auto... extents) { return ViewType("", extents...); },
                              extents_tuple);
            // return ViewType("",
            //                 static_cast<size_t>(tensor.pages()),
            //                 static_cast<size_t>(tensor.rows()),
            //                 static_cast<size_t>(tensor.cols()));
        }
        // device with ideal device layout
        else if constexpr (MemoryType == ViewMemoryType::OWNING)
        {
            // TODO: should this be using our template type?
            using ViewType = typename EquivalentView<ExecutionSpace,
                                                     typename ExecutionSpace::array_layout,
                                                     T>::type;

            return std::apply([&](auto... extents) { return ViewType("", extents...); },
                              extents_tuple);
            // return ViewType("", tensor.pages(), tensor.rows(), tensor.cols());
        }
    }

    // Specialization for Kokkos View
    template<typename T>
        requires IsKokkosView<T>
    static auto create_view(T& view)
    {
        // grab the extents with rank for the Kokkos views
        auto extents_tuple = [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            return std::make_tuple(static_cast<size_t>(view.extent(Is))...);
        }(std::make_index_sequence<T::rank()> {});

        // host layout may not be ideal for device
        if constexpr (MemoryType == ViewMemoryType::NONOWNING)
        {
            // NOTE: Eigen, by default uses "layoutleft" (meaning col-major order).
            // this differs from the default view which is LayoutRight.
            // originally the typename Kokkos::LayoutLeft was actually
            // typename ExecutationSpace::array_layout
            // TODO: a constexpr for if it's row major or not
            using ViewType = typename EquivalentView<ExecutionSpace,
                                                     typename ExecutionSpace::array_layout,
                                                     T>::type;

            //   typename EquivalentView<ExecutionSpace, typename Kokkos::LayoutLeft, T>::type;
            // this should return a shallow copy, but it ensures that things line up as expected
            // for our deep copies later
            return ViewType(view);
        }
        // tmp space on host with same layout as the device
        else if constexpr (MemoryType == ViewMemoryType::TMP)
        {
            // TODO: this is kind of dirty/hacky, as we should be passing the data in via
            // ArrayLayoutType instead of ScratchViewLayout, but this gets it working properly for
            // now
            // using ViewType = typename EquivalentView<ExecutionSpace, ScratchViewLayout, T>::type;
            // using ViewType = typename EquivalentView<ExecutionSpace, Kokkos::LayoutLeft,
            // T>::type;

            if constexpr (T::rank == 1)
            {
                using ViewType =
                  typename EquivalentView<ExecutionSpace, Kokkos::LayoutLeft, T>::type;
                // for scratch, if rank is 1, we don't need scratch, so it'll be 0 sized
                return ViewType("", 0);
            }
            else
            {
                // apply through iteration of the tuple
                using ViewType =
                  typename EquivalentView<ExecutionSpace, ScratchViewLayout, T>::type;
                return std::apply([&](auto... extents) { return ViewType("", extents...); },
                                  extents_tuple);
            }
        }
        // device with ideal device layout
        else if constexpr (MemoryType == ViewMemoryType::OWNING)
        {
            // TODO: should this be using our template type?
            using ViewType = typename EquivalentView<ExecutionSpace,
                                                     typename ExecutionSpace::array_layout,
                                                     T>::type;
            return std::apply([&](auto... extents) { return ViewType("", extents...); },
                              extents_tuple);
        }
    }

    // Creates view for a given execution space for a variadic list of data structures
    // (each needs a create_view specialization)

    template<typename Tuple, std::size_t... I>
    static auto create_views_helper(const Tuple& params_tuple,
                                    std::integer_sequence<std::size_t, I...>)
    {
        return std::make_tuple(
          Views<ExecutionSpace, MemoryType>::create_view(std::get<I>(params_tuple))...);
    }

    template<typename... ParameterTypes>
    static auto create_views_from_tuple(std::tuple<ParameterTypes...> params_tuple)
    {
        return create_views_helper(params_tuple,
                                   std::make_index_sequence<sizeof...(ParameterTypes)> {});
    }
};


template<typename T>
inline static auto create_views_inner_tuple(T& arg)
{
    // don't need scratch views when the host and device layouts are already the same
    return std::make_tuple(HostViewGenerator::create_view(arg),
                           DeviceViewGenerator::create_view(arg),
                           ScratchViewGenerator::create_view(arg));
}
template<typename Tp, std::size_t... I>
inline static auto create_views_outer_tuple(const Tp& t, std::integer_sequence<std::size_t, I...>)
{
    return std::make_tuple(create_views_inner_tuple(std::get<I>(t))...);
}
template<typename... T>
inline static auto create_views(std::tuple<T...> t)
{
    return create_views_outer_tuple(t, std::make_index_sequence<sizeof...(T)> {});
}

// utility function to return the type of a complex nested tuple
template<typename... T>
struct get_views_outer_tuple_type
{
    using type = decltype(create_views(std::make_tuple(std::declval<T>()...)));
};

// utility function to return the type of views tuple
template<typename T>
struct get_views_inner_tuple_type
{
    using type = decltype(create_views_inner_tuple(std::declval<T>()));
};


//
// ======== Tuple Transpose Functions
//

/**
 * @brief Inner-most function that creates the final set of tuples
 *
 * @tparam Idx The outer index (templated for consistent const at compile time)
 * @tparam T The main tuple type
 * @tparam I The iterated index type
 * @param t The tuple itself (passed by reference)
 * @return auto The new inner tuple that's transposed
 */
template<size_t Idx, typename T, std::size_t... I>
inline static auto repack_views_inner(const T& t, std::integer_sequence<std::size_t, I...>)
{
    // then just call std::make_tuple expanding out values
    // return std::make_tuple(get_v(I, Idx, t)...);
    return std::make_tuple(std::get<Idx>(std::get<I>(t))...);
}

/**
 * @brief Portion that iterates the tuples to get the proper indices
 *
 * @tparam Idx The outer index (templated for consistent const at compile time)
 * @tparam T The main tuple type, expanded so the compiler can count them
 * @param t The major tuple of tuples, templated to get the "sizeof"
 * @return auto The the tuple from repack_views_inner
 */
template<size_t Idx, typename... T>
inline static auto repack_views_outer(const std::tuple<T...>& t)
{
    // we now know what "device" index we're on, so we need to iterate over the total number
    // on the outer index
    return repack_views_inner<Idx>(t, std::make_index_sequence<sizeof...(T)> {});
}

/**
 * @brief Invert (transpose) the data views from Nx3 to 3xN for device isolation
 *
 * @tparam T The full deduced type of the tuples
 * @param t The N x 3 tuples, typically created via create_views
 * @return auto The 3 X N tuple of tuples of the views
 */
template<typename T>
inline static auto repack_views(const T& t)
{
    return std::make_tuple(repack_views_outer<0>(t),
                           repack_views_outer<1>(t),
                           repack_views_outer<2>(t));
}


//
// ======== View Helpers
//

/**
 * @brief Print a view
 *
 * @tparam ViewType The view type (should be deduced due to template complexity)
 * @param view The view to print
 */
template<typename ViewType>
inline void print_view(ViewType& view)
{
    if constexpr (ViewType::rank == 1)
    {
        for (size_t i = 0; i < view.extent(0); i++)
        {
            // Print all the elements on one line
            const double elem = view(i);
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    else if constexpr (ViewType::rank == 2)
    {
        for (size_t i = 0; i < view.extent(0); i++)
        {
            for (size_t j = 0; j < view.extent(1); j++)
            {
                std::cout << view(i, j) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    // else if (view.rank() == 3)
    // {
    //     for (size_t k = 0; k < view.extent(2); k++)
    //     {
    //         for (size_t j = 0; j < view.extent(1); j++)
    //         {
    //             for (size_t i = 0; i < view.extent(0); i++)
    //             {
    //                 // Print one row per line
    //                 const double elem = view(i, j, k);
    //                 std::cout << elem << " ";
    //             }
    //             std::cout << std::endl;
    //         }
    //     }
    // }
}
