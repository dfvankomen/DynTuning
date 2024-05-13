#pragma once

#include "common.hpp"
#include "range.hpp"
#include "kernel.hpp"
#include "Kokkos_Core.hpp"

struct FunctorVVM
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION
    void operator()(ViewsTuple views, const Index i) const
    {
        auto a = std::get<0>(views);
        auto b = std::get<1>(views);
        auto c = std::get<2>(views);
        c(i) = a(i) * b(i);
    }
};

template<typename... ParameterTypes>
inline auto KernelVVM(ParameterTypes&... data_params)
{
    auto name   = "vector-vector multiply";
    auto params = pack(data_params...);
    auto extent = range_extent(0, std::get<2>(params).size());
    return Kernel<1, FunctorVVM, ParameterTypes...>(name, params, extent);
}

template<typename KernelType, typename A, typename B, typename C>
inline void TestVVM(KernelType& k, A& a, B& b, C& c, DeviceSelector device)
{

    printf("\n%s\n", k.kernel_name_.c_str());

    auto& a_h = std::get<0>(k.data_views_host_);
    auto& b_h = std::get<1>(k.data_views_host_);
    auto& c_h = std::get<2>(k.data_views_host_);
    auto& a_d = std::get<0>(k.data_views_device_);
    auto& b_d = std::get<1>(k.data_views_device_);
    auto& c_d = std::get<2>(k.data_views_device_);
    
    //copy inputs from host to device
    if (device == DeviceSelector::DEVICE) {
        Kokkos::deep_copy(a_d, a_h);
        Kokkos::deep_copy(b_d, b_h);
    }
    
    // execute the kernel
    k(device);

    // copy output from device to host
    if (device == DeviceSelector::DEVICE) {
        Kokkos::deep_copy(c_h, c_d);
    }

    // check outputs
    size_t N = c.size();
    for (auto i = 0; i < N; i++) {
        printf("[%d] %f * %f = %f\n", i, a[i], b[i], c[i]);
        assert(c[i] == a[i] * b[i]);
    }

}