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

// This dummy range policy type is designed specifically to avoid
// kokkos errors. We're creating our own "rank 0" execution kernel,
// which gives the user complete access to Kokkos intrinsics, including
// the ability to write their own parallel-for and other such loops.
struct DummyRangePolicyType
{
    using value_type = std::uint64_t;
    value_type lower;
    value_type upper;

    DummyRangePolicyType(value_type low, value_type up)
      : lower(low)
      , upper(up) {};
};

// Used to rank to the corresponding RangePolicy type
template<int KernelRank,
         typename ExecutionSpace,
         typename LaunchBounds = Kokkos::LaunchBounds<0, 0>>
struct RangePolicy;

// 0-dimensional ranges (for giving the user full control over the kokkos kernel)
template<typename ExecutionSpace, typename LaunchBounds>
struct RangePolicy<0, ExecutionSpace, LaunchBounds>
{
    using type = DummyRangePolicyType;
};

// 1-dimensional ranges (MDRangePolicy does not support 1D ranges)
template<typename ExecutionSpace, typename LaunchBounds>
struct RangePolicy<1, ExecutionSpace, LaunchBounds>
{
    using type = Kokkos::RangePolicy<ExecutionSpace, LaunchBounds>;
};

// Dimensions > 1
template<int KernelRank, typename ExecutionSpace, typename LaunchBounds>
struct RangePolicy
{
    using type = Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<KernelRank>, LaunchBounds>;
};


//=============================================================================
// RangeExtent
//=============================================================================

using ArrayIndex = std::uint64_t;

// Used to map rank to the corresponding extent for ranges type
template<int KernelRank>
struct RangeExtent;

// 0-dimensional ranges (for when the user needs control in the kernels)
template<>
struct RangeExtent<0>
{
    using value_type = ArrayIndex;
    value_type lower;
    value_type upper;
};

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

inline RangeExtent<0> range_extent()
{
    return { 0, 0 };
}

inline RangeExtent<1> range_extent(const ArrayIndex& lower, const ArrayIndex& upper)
{
    return { lower, upper };
}

inline RangeExtent<2> range_extent(const Kokkos::Array<ArrayIndex, 2>& lower,
                                   const Kokkos::Array<ArrayIndex, 2>& upper)
{
    return { lower, upper };
}

// for some reason, this doesn't actually get selected via auto, so it needs to be by hand
inline RangeExtent<3> range_extent(const Kokkos::Array<ArrayIndex, 3>& lower,
                                   const Kokkos::Array<ArrayIndex, 3>& upper)
{
    return { lower, upper };
}
