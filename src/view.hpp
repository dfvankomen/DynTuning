#pragma once

#include "Kokkos_Core.hpp"

// view memory type options
enum class ViewMemoryType
{
    NONOWNING,
    TMP,
    OWNING
};

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

/*
#ifdef USE_EIGEN
template<typename EigenT, typename KokkosLayout>
concept IsLayoutSame =
  (std::is_same_v<KokkosLayout, Kokkos::LayoutRight> && std::decay_t<EigenT>::IsRowMajor == 1) ||
  (std::is_same_v<KokkosLayout, Kokkos::LayoutLeft> && std::decay_t<EigenT>::IsRowMajor == 0);
#endif
*/

#ifdef USE_EIGEN
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
#endif

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

#ifdef USE_EIGEN
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
            return ViewType("", 0, 0);

        // device with ideal device layout
        } else if constexpr (MemoryType == ViewMemoryType:: OWNING) {
            using ViewType = typename EquivalentView<ExecutionSpace, Kokkos::LayoutLeft, T>::type;
            return ViewType("", matrix.rows(), matrix.cols());
        }
    }
#endif

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