#pragma once

#include "Kokkos_Core.hpp"

// Note: TODO think about how to make range policy generic to work for:
// 1) Matrix-vector operation (different range)
// 2) Matrix-matrix operation
// 3) Vector-vector operation of same size
// 4) vector-vector opertion with difference sizes (convolution)

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


//=============================================================================
// Helpers
//=============================================================================

inline RangeExtent<1> range_extent(const ArrayIndex& lower, const ArrayIndex& upper)
{
    return { lower, upper };
}

inline RangeExtent<2> range_extent(const Kokkos::Array<ArrayIndex, 2>& lower,
                            const Kokkos::Array<ArrayIndex, 2>& upper)
{
    return { lower, upper };
}