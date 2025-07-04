#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"

struct FunctorVectorDot_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i) const
    {
        auto a = std::get<0>(views);
        auto b = std::get<1>(views);
        auto c = std::get<2>(views);

        c(i) = a(i) * b(i);
    }
};

struct FunctorVectorDot_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i) const
    {
        // get inputs a and b
        auto a = std::get<0>(views);
        auto b = std::get<1>(views);
        // get the output c
        auto c = std::get<2>(views);
        // perform the math
        c(i) = a(i) * b(i);
    }
};

/*
template <typename T>
inline auto to_const_ref(T& arg, bool flag, int i, int j)
{
    if (flag) {
        printf("const! %d %d %d\n", i, j, flag);
        return std::as_const(arg);
    } else {
        printf("nonconst %d %d %d\n", i, j, flag);
        return arg;
    }
}
*/

#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))

template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelVectorDot(KernelOptions& options, ParameterTypes&... data_views)
{
    auto name = "vector-vector multiply";

    auto is_const = kernel_io_map(data_views...);

    // WARNING, remapping loses const! Fix this later...
    // transpose the views tuple
    // views_: i = variable, j = device
    // views:  i = device, j = variable
    auto views_ = pack(data_views...);

    auto views = repack_views(views_);

    // set the extent based on the host view of the output
    auto out    = std::get<2>(std::get<0>(views));
    auto extent = range_extent(0, out.extent(0));

    // generate the policy collection based on user hyperparameters
    auto full_policy_collection =
      make_policy_from_hyperparameters<1, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();


    // create the kernel
    return Kernel<1,
                  FunctorVectorDot_Host,
                  FunctorVectorDot_Device,
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
inline void TestVectorDot(KernelType& k, std::vector<double>& a, std::vector<double>& b,
std::vector<double>& c)
{
    std::vector<DeviceSelector> devices = k.options_.devices;

    for (DeviceSelector device : devices) {

        std::cout << "\n" << k.kernel_name_ << " (" << device << ")" << std::endl;

        auto& a_h = std::get<0>(std::get<0>(k.data_views_));
        auto& b_h = std::get<1>(std::get<0>(k.data_views_));
        auto& c_h = std::get<2>(std::get<0>(k.data_views_));

        auto& a_d = std::get<0>(std::get<1>(k.data_views_));
        auto& b_d = std::get<1>(std::get<1>(k.data_views_));
        auto& c_d = std::get<2>(std::get<1>(k.data_views_));

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
*/