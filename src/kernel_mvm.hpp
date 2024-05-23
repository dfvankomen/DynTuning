#pragma once

#include "common.hpp"
#include "range.hpp"
#include "kernel.hpp"
#include "Kokkos_Core.hpp"
#include "Eigen"

//https://github.com/kokkos/kokkos-tutorials/blob/main/Exercises/mdrange/Solution/exercise_mdrange_solution.cpp

struct FunctorMVM_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION
    void operator()(ViewsTuple views, const Index i, const Index j) const
    {

        auto A = std::get<0>(views);
        auto x = std::get<1>(views);
        auto b = std::get<2>(views);
        b(i) += A(i, j) * x(j);

//printf("%f = %f * %f\n", A(i, j) * x(j), A(i,j), x(j));

    }
};

struct FunctorMVM_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION
    void operator()(ViewsTuple views, const Index i, const Index j) const
    {

        auto A = std::get<0>(views);
        auto x = std::get<1>(views);
        auto b = std::get<2>(views);
        b(j) += A(j, i) * x(i);

//printf("%f = %f * %f\n", A(i, j) * x(j), A(i,j), x(j));

    }
};

template<typename... ParameterTypes>
inline auto KernelMVM(KernelOptions& options, ParameterTypes&... data_views)
{
    auto name       = "matrix-vector multiply";

    auto is_const = kernel_io_map(data_views...);

    // WARNING, remapping loses const! Fix this later...
    // transpose the views tuple
    // views_: i = variable, j = device
    // views:  i = device, j = variable
    auto views_ = pack(data_views...);

    auto views = std::make_tuple(
        std::make_tuple(
            get_v(0, 0, views_),
            get_v(1, 0, views_),
            get_v(2, 0, views_)
        ),
        std::make_tuple(
            get_v(0, 1, views_),
            get_v(1, 1, views_),
            get_v(2, 1, views_)
        ),
        std::make_tuple(
            get_v(0, 2, views_),
            get_v(1, 2, views_),
            get_v(2, 2, views_)
        )
    );

    // set the extent based on the host view of the matrix
    auto out        = std::get<0>(std::get<0>(views));
    unsigned long N = out.extent(0);
    unsigned long M = out.extent(1);
    auto extent     = range_extent({ 0, 0 }, { N, M });
    return Kernel<2, FunctorMVM_Host, FunctorMVM_Device, decltype(views), decltype(is_const)>(
        name, views, is_const, extent, options
    );
}

/*
template<typename KernelType>
inline void TestMVM(KernelType& k, Eigen::MatrixXd& a, std::vector<double>& b, std::vector<double>& c)
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