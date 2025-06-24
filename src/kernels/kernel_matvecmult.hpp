#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"

#include <Eigen/Eigen>

// https://github.com/kokkos/kokkos-tutorials/blob/main/Exercises/mdrange/Solution/exercise_mdrange_solution.cpp

// TODO: this kernel requires parallel reduce!

struct FunctorMatVecMult_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        // Input matrix A, input vector x, output vector b
        auto A = std::get<0>(views);
        auto x = std::get<1>(views);
        auto b = std::get<2>(views);

        // Atomic addition for race condition
        Kokkos::atomic_add(&b(i), A(i, j) * x(j));
    }
};

struct FunctorMatVecMult_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        // Input matrix A, input vector x, output vector b
        auto A = std::get<0>(views);
        auto x = std::get<1>(views);
        auto b = std::get<2>(views);

        // Atomic addition for race condition
        Kokkos::atomic_add(&b(i), A(i, j) * x(j));
    }
};

#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))

template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelMatVecMult(KernelOptions& options, ParameterTypes&... data_views)
{
    constexpr int KernelRank = 2;

    auto name = "matrix-vector multiply";

    auto is_const = kernel_io_map(data_views...);

    // WARNING, remapping loses const! Fix this later...
    // transpose the views tuple
    // views_: i = variable, j = device
    // views:  i = device, j = variable
    auto views_ = pack(data_views...);

    auto views = repack_views(views_);

    // set the extent based on the host view of the matrix
    auto out                       = std::get<0>(std::get<0>(views));
    unsigned long N                = out.extent(0);
    unsigned long M                = out.extent(1);
    RangeExtent<KernelRank> extent = range_extent(Kokkos::Array<std::uint64_t, 2> { 0, 0 },
                                                  Kokkos::Array<std::uint64_t, 2> { N, M });

    // execution policy tuple to store the different types of policies
    // TODO: probably don't need to specify the device for this function
    // auto full_policy_collection = create_range_policy_device<2, Kokkos::KOKKOS_DEVICE>(extent);
    // auto policy_names = create_range_policy_device_collection();

    // TODO: user can adjust the policy via a similar method:
    // TODO: probably don't need to specify the device for this function
    // auto full_policy_collection =
    //   create_range_policy_device<2, Kokkos::KOKKOS_DEVICE, 32, 150, 8, 1, 10, 3>(extent);

    // generate the policy collection based on user hyperparameters
    auto full_policy_collection =
      make_policy_from_hyperparameters<KernelRank, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(
        extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();

    return Kernel<KernelRank,
                  FunctorMatVecMult_Host,
                  FunctorMatVecMult_Device,
                  decltype(views),
                  decltype(is_const),
                  decltype(full_policy_collection),
                  decltype(policy_names)>(name,
                                          views,
                                          is_const,
                                          extent,
                                          options,
                                          full_policy_collection,
                                          policy_names);
}

/*
template<typename KernelType>
inline void TestMatVecMult(KernelType& k, Eigen::MatrixXd& a, std::vector<double>& b,
std::vector<double>& c)
{

    std::vector<DeviceSelector> devices = k.options_.devices;

    for (DeviceSelector device : devices) {

        std::cout << "\n" << k.kernel_name_ << "(" << device << ")" << std::endl;

        auto& a_h = std::get<0>(std::get<0>(k.data_views_));
        auto& b_h = std::get<1>(std::get<0>(k.data_views_));
        auto& c_h = std::get<2>(std::get<0>(k.data_views_));

        auto& a_t = std::get<0>(std::get<2>(k.data_views_));
        auto& b_t = std::get<1>(std::get<2>(k.data_views_));
        auto& c_t = std::get<2>(std::get<2>(k.data_views_));

        auto& a_d = std::get<0>(std::get<1>(k.data_views_));
        auto& b_d = std::get<1>(std::get<1>(k.data_views_));
        auto& c_d = std::get<2>(std::get<1>(k.data_views_));

        //copy inputs from host to host tmp space to convert layout, then to device
        if (device == DeviceSelector::DEVICE) {
            Kokkos::deep_copy(a_t, a_h); Kokkos::deep_copy(a_d, a_t);
            Kokkos::deep_copy(b_d, b_h);
        }

        // execute the kernel
        k(device);

        // copy output from device to host tmp space then to host
        if (device == DeviceSelector::DEVICE) {
            Kokkos::deep_copy(c_h, c_d);
        }

        // check outputs
        unsigned long N = a.rows();
        unsigned long M = a.cols();
        for (auto i = 0; i < N; i++) {
            std::cout << "[" << i << "] ", c[i], " =";
            double c_ = 0.0;
            for (auto j = 0; j < M; j++) {
                if (j > 0) std::cout << " +";
                std::cout << " (" << a(i,j) << "*" << b[j] << ")";
                c_ += a(i,j) * b[j];
            }
            std::cout << " = " << c_ << " = " << c[i] << std::endl;
            //assert(c[i] == c_);
        }
//b(i) += A(i, j) * x(j);
//c(i) += a(i, j) * b(j);
    }
}
*/
