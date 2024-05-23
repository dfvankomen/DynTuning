#pragma once

#include "Kokkos_Core.hpp"

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
using HostViewSpace     = HostExecutionSpace;
using DeviceViewSpace   = DeviceExecutionSpace;
using ScratchViewSpace  = HostViewSpace;

// layout
using HostViewLayout    = typename HostExecutionSpace::array_layout;
using DeviceViewLayout  = Kokkos::LayoutLeft;
using ScratchViewLayout = DeviceViewLayout;

// generator
template<typename ExecutionSpace, ViewMemoryType MemoryType> struct Views;
using HostViewGenerator    = Views<HostViewSpace,    ViewMemoryType::NONOWNING>;
using DeviceViewGenerator  = Views<DeviceViewSpace,  ViewMemoryType::OWNING>;
using ScratchViewGenerator = Views<ScratchViewSpace, ViewMemoryType::TMP>;

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

//=============================================================================
// EquivalentView
//=============================================================================

//https://kokkos.org/kokkos-core-wiki/API/core/KokkosConcepts.html
//https://kokkos.org/kokkos-core-wiki/API/core/view/view.html

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
    //using type = Kokkos::View<value_type*,
    //                          typename ExecutionSpace::array_layout,
    //                          typename ExecutionSpace::memory_space>;

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

//=============================================================================
// Views
//=============================================================================

template<typename ExecutionSpace, ViewMemoryType MemoryType>
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
        if constexpr (MemoryType == ViewMemoryType::NONOWNING) {
            using ViewType = typename EquivalentView<ExecutionSpace, typename ExecutionSpace::array_layout, T>::type;
            //using ViewType = typename EquivalentView<ExecutionSpace, T>::type;
            return ViewType(const_cast<ViewType::value_type*>(vector.data()), vector.size());

        // vectors don't need tmp space
        } else if constexpr (MemoryType == ViewMemoryType::TMP) {
            using ViewType = typename EquivalentView<ExecutionSpace, Kokkos::LayoutLeft, T>::type;
            return ViewType("", 0);

        // device with ideal device layout
        } else if constexpr (MemoryType == ViewMemoryType:: OWNING) {
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
        if constexpr (MemoryType == ViewMemoryType::NONOWNING) {
            using ViewType = typename EquivalentView<ExecutionSpace, typename ExecutionSpace::array_layout, T>::type;
            //using ViewType = typename EquivalentView<ExecutionSpace, T>::type;
            return ViewType(const_cast<ViewType::value_type*>(matrix.data()),
                static_cast<size_t>(matrix.rows()), static_cast<size_t>(matrix.cols()));

        // tmp space on host with same layout as the device
        } else if constexpr (MemoryType == ViewMemoryType::TMP) {
            using ViewType = typename EquivalentView<ExecutionSpace, Kokkos::LayoutLeft, T>::type;
            return ViewType("", static_cast<size_t>(matrix.rows()), static_cast<size_t>(matrix.cols()));

        // device with ideal device layout
        } else if constexpr (MemoryType == ViewMemoryType:: OWNING) {
            using ViewType = typename EquivalentView<ExecutionSpace, Kokkos::LayoutLeft, T>::type;
            return ViewType("", matrix.rows(), matrix.cols());
        }
    }

    // Creates view for a given execution space for a variadic list of data structures
    // (each needs a create_view specialization)

    template<typename Tuple, std::size_t... I>
    static auto create_views_helper(const Tuple& params_tuple,
                                    std::integer_sequence<std::size_t, I...>)
    {
        return std::make_tuple(Views<ExecutionSpace, MemoryType>::create_view(std::get<I>(params_tuple))...);
    }

    template<typename... ParameterTypes>
    static auto create_views_from_tuple(std::tuple<ParameterTypes...> params_tuple)
    {
        return create_views_helper(params_tuple,
                                   std::make_index_sequence<sizeof...(ParameterTypes)> {});
    }

};


template <typename T>
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
template <typename... T>
struct get_views_outer_tuple_type {
    using type = decltype(create_views(std::make_tuple(std::declval<T>()...)));
};

// utility function to return the type of views tuple
template <typename T>
struct get_views_inner_tuple_type {
    using type = decltype(create_views_inner_tuple(std::declval<T>()));
};



template <typename ViewType>
inline void print_view(ViewType& view) {
    if (view.rank() == 1) {
        for (size_t i = 0; i < view.extent(0); i++) {
            // Print all the elements on one line
            const double elem = view(i);
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    } /*else if (view.rank() == 2) {
        for (size_t j = 0; j < view.extent(1); j++) {
            for (size_t i = 0; i < view.extent(0); i++) {
                // Print one row per line
                const double elem = view(i,j);
                std::cout << elem << " ";
            }
            std::cout << std::endl;
        }
    } else if (view.rank() == 3) {
        for (size_t k = 0; k < view.extent(2); k++) {
            for (size_t j = 0; j < view.extent(1); j++) {
                for (size_t i = 0; i < view.extent(0); i++) {
                    // Print one row per line
                    const double elem = view(i,j,k);
                    std::cout << elem << " ";
                }
                std::cout << std::endl;
            }
        }
    }*/
}