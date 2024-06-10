#pragma once

#include <iostream>
#include <numeric>
#include <map>

//#ifdef USE_EIGEN
#include "Eigen"
//#endif

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

inline std::ostream& operator<<(std::ostream& out, const DeviceSelector value)
{
    std::map<DeviceSelector, std::string> m;
    m[DeviceSelector::AUTO]   = "a";
    m[DeviceSelector::HOST]   = "h";
    m[DeviceSelector::DEVICE] = "d";
    return out << m[value];
}

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


// base case for recursion of find_tuple
template<typename LambdaType, std::size_t I = 0, typename... T>
inline typename std::enable_if<I == sizeof...(T), void>::type
find_tuple(const std::tuple<T...>& t, std::size_t idx, const LambdaType& lambda)
{
    // if we get to this point, then there's no index here, so we 
    throw std::out_of_range("Could not find index " + std::to_string(idx) + " in requested tuple. The value is too large!");
}

// all other cases, it's a recursive template that iterates through and finds the item in the tuple
// note that the lambda must have the right typing cast when calling it, as it needs to receive itself
template<typename LambdaType, std::size_t I = 0, typename... T>
inline typename std::enable_if<I < sizeof...(T), void>::type
find_tuple(const std::tuple<T...>& t, std::size_t idx, const LambdaType& lambda)
{

    // if we've found the index, we can execute the lambda and return out
    if (I == idx){
        auto &elem = std::get<I>(t);
        lambda(elem);
    }
    // otherwise we continue to recurse through until we find the lambda
    else {
        find_tuple<LambdaType, I+1, T...>(t, idx, lambda);
    }
}

/*
// utility function for getting the ith item of a tuple
template<std::size_t I = 0, typename... T>
inline typename std::enable_if<I == sizeof...(T), void>::type
get_tuple_item(int i, const std::tuple<T...>& t) {}
template<std::size_t I = 0, typename... T>
inline typename std::enable_if<I < sizeof...(T), void>::type
get_tuple_item(int i, const std::tuple<T...>& t)
{
    if (i == I)
        return std::get<I>(t);
    else
        return get_tuple_item<I + 1, T...>(i, t);
}
*/


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
            std::cout << "device = " << arg << std::endl;
            break;
        }
    }
    if (device == DeviceSelector::AUTO)
        std::cout << "device = " << "auto" << std::endl;
    return device;
}

inline int set_N(int argc, char* argv[])
{
    int N = 5;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--N=") == 0) {
            N = std::atoi(arg.substr(4).c_str());
            break;
        }
    }
    std::cout << "N = " << N << std::endl;
    return N;
}

inline bool set_reordering(int argc, char* argv[])
{
    bool flag = false;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--reordering") == 0) {
            flag = true;
            break;
        }
    }
    std::string s(flag ? "true" : "false");
    std::cout << "reordering = " << s << std::endl;
    return flag;
}

inline int set_initialize(int argc, char* argv[])
{
    bool flag = true;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--no-initialize") == 0) {
            flag = false;
            break;
        }
    }
    std::string s(flag ? "true" : "false");
    std::cout << "initialize = " << s << std::endl;
    return flag;
}

inline int set_num_sims(int argc, char* argv[])
{
    int N = 5;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--num_sims=") == 0) {
            N = std::atoi(arg.substr(11).c_str());
            break;
        }
    }
    std::cout << "num_sims = " << N << std::endl;
    return N;
}

inline int set_num_chain_runs(int argc, char* argv[])
{
    int N = 1;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--chain_runs=") == 0) {
            N = std::atoi(arg.substr(13).c_str());
            break;
        }
    }
    std::cout << "chain_runs = " << N << std::endl;
    return N;
}


#ifdef DYNTUNE_SINGLE_CHAIN_RUN
inline unsigned int set_single_chain_run(int argc, char* argv[])
{
    unsigned int N = 0;
    for (int i = 0; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("--single_chain=") == 0) {
            N = std::atoi(arg.substr(15).c_str());
            break;
        }
    }
    std::cout << "single_chain = " << N << std::endl;
    return N;
}
#endif

template <typename T>
inline void print_is_reference(const T& arg) {
    if constexpr (std::is_reference_v<T>) {
        std::cout << "The argument is a reference." << std::endl;
    } else {
        std::cout << "The argument is not a reference." << std::endl;
    }
}