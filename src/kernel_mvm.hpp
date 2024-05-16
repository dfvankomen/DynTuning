#pragma once

#include "common.hpp"
#include "range.hpp"
#include "Kokkos_Core.hpp"
#include "Eigen"

struct FunctorMVM
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION
    void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        auto A = std::get<0>(views);
        auto x = std::get<1>(views);
        auto b = std::get<2>(views);
        b(i) += A(i, j) * x(j);
    }
};

template<typename... ParameterTypes>
inline auto KernelMVM(KernelOptions& options, ParameterTypes&... data_params)
{
    auto name       = "matrix-vector multiply";
    auto params     = pack(data_params...);
    unsigned long N = std::get<0>(params).rows();
    unsigned long M = std::get<0>(params).cols();
    auto extent     = range_extent({ 0, 0 }, { N, M });
    return Kernel<2, FunctorMVM, ParameterTypes...>(name, params, extent, options);
}

template<typename KernelType>
inline void TestMVM(KernelType& k)
{
    auto& a = std::get<0>(k.data_params_);
    auto& b = std::get<1>(k.data_params_);
    auto& c = std::get<2>(k.data_params_);

    std::vector<DeviceSelector> devices = k.options_.devices;

    for (DeviceSelector device : devices) {

        std::cout << "\n" << k.kernel_name_ << "(" << device << ")" << std::endl;
        
        auto& a_h = std::get<0>(k.data_views_host_);
        auto& b_h = std::get<1>(k.data_views_host_);
        auto& c_h = std::get<2>(k.data_views_host_);
        auto& a_t = std::get<0>(k.data_views_tmp_);
        auto& b_t = std::get<1>(k.data_views_tmp_);
        auto& c_t = std::get<2>(k.data_views_tmp_);
        auto& a_d = std::get<0>(k.data_views_device_);
        auto& b_d = std::get<1>(k.data_views_device_);
        auto& c_d = std::get<2>(k.data_views_device_);
        
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
        unsigned long N = c.rows();
        unsigned long M = c.cols();
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

    }
}