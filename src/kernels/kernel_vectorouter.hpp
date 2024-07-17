#pragma once

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"

#include <stdexcept>

struct FunctorVectorOuter_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        auto a = std::get<0>(views);
        auto b = std::get<1>(views);
        auto C = std::get<2>(views);

        C(i, j) = a(i) * b(j);
    }
};

struct FunctorVectorOuter_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views, const Index i, const Index j) const
    {
        auto a = std::get<0>(views);
        auto b = std::get<1>(views);
        auto C = std::get<2>(views);

        C(i, j) = a(i) * b(j);
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

template<typename... ParameterTypes>
inline auto KernelVectorOuter(KernelOptions& options, ParameterTypes&... data_views)
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
    auto out = std::get<2>(std::get<0>(views));

    unsigned long N = out.extent(0);
    unsigned long M = out.extent(1);

    // do a verification with the extent 0 of the input vectors
    if (std::get<0>(std::get<0>(views)).extent(0) != N)
        throw std::invalid_argument(
          "Vector A's length doesn't match the first dimension of the output matrix!");

    if (std::get<1>(std::get<0>(views)).extent(0) != M)
        throw std::invalid_argument(
          "Vector B's length doesn't match the second dimension of the output matrix!");

    auto extent = range_extent({ 0, 0 }, { N, M });

    // then build up the policy collection
    auto full_policy_collection = create_range_policy_device<2>(extent);
    auto policy_names           = create_range_policy_device_collection();

    // TODO: user can adjust the policy via a similar method:
    // TODO: probably don't need to specify the device for this function
    // using KernelLinspaceParameters = LinspaceOptions<32, 512, 2, 1, 10, 2>;
    // auto full_policy_collection =
    //   create_range_policy_device<2, Kokkos::KOKKOS_DEVICE, KernelLinspaceParameters>(extent);
    // auto policy_names = create_range_policy_device_collection<KernelLinspaceParameters>();

    // create the kernel
    return Kernel<2,
                  FunctorVectorOuter_Host,
                  FunctorVectorOuter_Device,
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
inline void TestVectorOuter(KernelType& k, std::vector<double>& a, std::vector<double>& b,
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