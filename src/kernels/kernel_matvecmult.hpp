#pragma once

#include "Eigen"
#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"

// https://github.com/kokkos/kokkos-tutorials/blob/main/Exercises/mdrange/Solution/exercise_mdrange_solution.cpp

// TODO: this kernel requires parallel reduce!

struct FunctorMatVecMult_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {

        auto A = std::get<0>(views);
        auto x = std::get<1>(views);
        auto b = std::get<2>(views);

        Kokkos::atomic_add(&b(i), A(i, j) * x(j));

        // b(i) += A(i, j) * x(j);

        // need to return b(i) instead of trying to update automatically!

        // printf("%f = %f * %f\n", A(i, j) * x(j), A(i,j), x(j));
    }
};

struct FunctorMatVecMult_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        // NOTE: race condition because we're trying to update b(i) at the same time

        auto A = std::get<0>(views);
        auto x = std::get<1>(views);
        auto b = std::get<2>(views);

        // for the CPUj side, is atomic add necessary?
        // my gut says no, but that depends on the backend that's working
        Kokkos::atomic_add(&b(i), A(i, j) * x(j));
        // b(i) += A(i, j) * x(j);

        // need to return b(i) instead of trying to update automatically!

        // printf("( %d %d ) %f = %f * %f\n", i, j, A(i, j) * x(j), A(i, j), x(j));
    }
};

#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))

template<typename... ParameterTypes>
inline auto KernelMatVecMult(KernelOptions& options, ParameterTypes&... data_views)
{
    auto name = "matrix-vector multiply";

    auto is_const = kernel_io_map(data_views...);

    // WARNING, remapping loses const! Fix this later...
    // transpose the views tuple
    // views_: i = variable, j = device
    // views:  i = device, j = variable
    auto views_ = pack(data_views...);

    auto views = repack_views(views_);

    // set the extent based on the host view of the matrix
    auto out        = std::get<0>(std::get<0>(views));
    unsigned long N = out.extent(0);
    unsigned long M = out.extent(1);
    auto extent     = range_extent({ 0, 0 }, { N, M });

    // execution policy tuple to store the different types of policies
    // TODO: probably don't need to specify the device for this function
    auto full_policy_collection =
      create_range_policy_device<2, Kokkos::KOKKOS_DEVICE, 32, 150, 2, 1, 10, 2>(extent);

    // TODO: user can adjust the policy via a similar method:
    // TODO: probably don't need to specify the device for this function
    // auto full_policy_collection =
    //   create_range_policy_device<2, Kokkos::KOKKOS_DEVICE, 32, 150, 8, 1, 10, 3>(extent);

    return Kernel<2,
                  FunctorMatVecMult_Host,
                  FunctorMatVecMult_Device,
                  decltype(views),
                  decltype(is_const),
                  decltype(full_policy_collection)>(name,
                                                    views,
                                                    is_const,
                                                    extent,
                                                    options,
                                                    full_policy_collection);
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