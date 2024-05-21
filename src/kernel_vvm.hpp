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
inline auto KernelVVM(KernelOptions& options, ParameterTypes&... data_views)
{
    auto name   = "vector-vector multiply";

    // transpose the views tuple
    auto views_ = pack(data_views...);
    auto views = std::make_tuple(
        std::make_tuple(
            std::get<0>(std::get<0>(views_)),
            std::get<0>(std::get<1>(views_)),
            std::get<0>(std::get<2>(views_))
        ),
        std::make_tuple(
            std::get<1>(std::get<0>(views_)),
            std::get<1>(std::get<1>(views_)),
            std::get<1>(std::get<2>(views_))
        ),
        std::make_tuple(
            std::get<2>(std::get<0>(views_)),
            std::get<2>(std::get<1>(views_)),
            std::get<2>(std::get<2>(views_))
        )
    );

    // set the extent based on the host view of the output
    auto out    = std::get<2>(std::get<0>(views));
    auto extent = range_extent(0, out.extent(0));

    // create the kernel
    return Kernel<1, FunctorVVM, decltype(views)>(
        name, views, extent, options
    );
}

template<typename KernelType>
inline void TestVVM(KernelType& k, std::vector<double>& a, std::vector<double>& b, std::vector<double>& c)
{
    std::vector<DeviceSelector> devices = k.options_.devices;

    for (DeviceSelector device : devices) {

        std::cout << "\n" << k.kernel_name_ << " (" << device << ")" << std::endl;

        auto& a_h = std::get<0>(std::get<0>(k.data_views_));
        auto& b_h = std::get<0>(std::get<1>(k.data_views_));
        auto& c_h = std::get<0>(std::get<2>(k.data_views_));

        auto& a_d = std::get<1>(std::get<0>(k.data_views_));
        auto& b_d = std::get<1>(std::get<1>(k.data_views_));
        auto& c_d = std::get<1>(std::get<2>(k.data_views_));
        
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

}