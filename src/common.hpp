#pragma once

#include <iostream>
#include <numeric>

#ifdef USE_EIGEN
#include "Eigen"
#endif

// ensure KOKKOS_HOST is set
#ifndef KOKKOS_HOST
#define KOKKOS_HOST Serial
#endif

// ensure KOKKOS_DEVICE is set
#ifndef KOKKOS_DEVICE
#define KOKKOS_DEVICE Cuda
#endif

// device selector options
enum class DeviceSelector
{
    AUTO,
    HOST,
    DEVICE
};

/*
// simple macro for timing a function
#define TIMING(k, f)                                 \
    {                                                \
        Kokkos::Timer timer;                         \
        timer.reset();                               \
        f;                                           \
        double kernel_time = timer.seconds();        \
        printf("%s: %.6f\n", k.name(), kernel_time); \
    }
*/

/// @brief Packs a list of references to class instances into a tuple
/// @tparam ...ParameterTypes
/// @param ...params
/// @return
template<typename... ParameterTypes>
inline auto pack(ParameterTypes&... params) -> std::tuple<ParameterTypes&...>
{
    // note: std::forward loses the reference qualifer... check into this later
    return std::make_tuple(std::ref(params)...);
}

// utility function for iterating over a tuple of unknown length
template<typename LambdaType, std::size_t I = 0, typename... T>
inline typename std::enable_if<I == sizeof...(T), void>::type
iter_tuple(const std::tuple<T...>& t, const LambdaType& lambda)
{
}
template<typename LambdaType, std::size_t I = 0, typename... T>
inline typename std::enable_if <I < sizeof...(T), void>::type
iter_tuple(const std::tuple<T...>& t, const LambdaType& lambda)
{
    auto& elem = std::get<I>(t);
    lambda(I, elem);
    iter_tuple<LambdaType, I + 1, T...>(t, lambda);
}

inline DeviceSelector set_device(int argc, char* argv[])
{
    DeviceSelector device = DeviceSelector::AUTO;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--device=") == 0) {
            arg = arg.substr(9);
            const char *s = &arg.c_str()[0];
            if (strcmp(s, "device") == 0) {
                device = DeviceSelector::DEVICE;
            } else if (strcmp(s, "host") == 0) {
                device = DeviceSelector::HOST;
            }
            std::cout << "Device = " << arg << std::endl;
            break;
        }
    }
    if (device == DeviceSelector::AUTO)
        std::cout << "Device = " << "auto" << std::endl;
    return device;
}

inline int set_N(int argc, char* argv[])
{
    int N = 5;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--N=") == 0) {
            N = std::atoi(arg.substr(4).c_str());
        }
    }
    std::cout << "N = " << N << std::endl;
    return N;
}