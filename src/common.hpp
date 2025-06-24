#pragma once

#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>

// #ifdef USE_EIGEN
#include <Eigen/Eigen>
// #endif

// ensure KOKKOS_HOST is set
#ifndef KOKKOS_HOST
#define KOKKOS_HOST Serial
#endif

// ensure KOKKOS_DEVICE is set
#ifndef KOKKOS_DEVICE
#define KOKKOS_DEVICE Cuda
#endif

/**
 * @brief A useful helper function for debugging classes without type expansion
 *
 * Use this with a print statement or stream to see the output.
 * Taken from https://stackoverflow.com/posts/59522794/revisions
 *
 * @tparam T Any template type
 * @return const char* Output string that prints the full name of the function
 */
template<typename T>
const char* prettyprint_function_type()
{
#ifdef _MSC_VER
    return __FUNCSIG__;
#else
    return __PRETTY_FUNCTION__;
#endif
}

/**
 * @brief A convenience typedef for an Eigen2D matrix with row major order
 *
 */
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> DynMatrix2D;

/**
 * @brief Device selection enum for accessing AUTO, HOST, or DEVICE
 *
 */
enum class DeviceSelector
{
    AUTO,
    HOST,
    DEVICE
};

/**
 * @brief Stream operator for DeviceSelector
 *
 * @param out Output stream
 * @param value The value of the selector
 * @return std::ostream& Outputs
 */
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

/**
 * @brief Packs a list of references to class instances into a tuple
 *
 * @tparam ParameterTypes Inferred parameter types
 * @param params The parameters that should be packed up
 * @return std::tuple<ParameterTypes&...> Tuple containing the parameters
 */
template<typename... ParameterTypes>
inline auto pack(ParameterTypes&... params) -> std::tuple<ParameterTypes&...>
{
    // note: std::forward loses the reference qualifer... check into this later
    return std::make_tuple(std::ref(params)...);
}

// utility function for iterating over a tuple of unknown length
/**
 * @brief Utility function for iterating over a tuple of unknown length
 *
 * @tparam LambdaType Inferred Lambda type
 * @tparam I Inferred index
 * @tparam T Inferred type inside the tuple
 * @param t Input tuple
 * @param lambda Lambda that should be run
 * @return std::enable_if<I == sizeof...(T), void>::type
 */
template<typename LambdaType, std::size_t I = 0, typename... T>
inline typename std::enable_if<I == sizeof...(T), void>::type iter_tuple(const std::tuple<T...>& t,
                                                                         const LambdaType& lambda)
{
}

/**
 * @brief Function that actually performs the tuple iteration
 *
 * @tparam LambdaType Inferred Lambda type
 * @tparam I Inferred tuple index
 * @tparam T Inferred types inside the tuple
 */
template<typename LambdaType, std::size_t I = 0, typename... T>
  inline typename std::enable_if <
  I<sizeof...(T), void>::type iter_tuple(const std::tuple<T...>& t, const LambdaType& lambda)
{
    auto& elem = std::get<I>(t);
    lambda(I, elem);
    iter_tuple<LambdaType, I + 1, T...>(t, lambda);
}


// base case for recursion of find_tuple
/**
 * @brief Portion of find_tuple if the index wasn't found
 *
 * @tparam LambdaType Inferred Lambda type
 * @tparam I Inferred tuple index
 * @tparam T Inferred types inside the tuple
 * @param t Tuple being searched
 * @param idx Index de1sired
 * @param lambda Lambda function to run
 */
template<typename LambdaType, std::size_t I = 0, typename... T>
inline typename std::enable_if<I == sizeof...(T), void>::type find_tuple(const std::tuple<T...>& t,
                                                                         std::size_t idx,
                                                                         const LambdaType& lambda)
{
    // if we get to this point, then there's no index here, so we
    throw std::out_of_range("Could not find index " + std::to_string(idx) +
                            " in requested tuple. The value is too large!");
}

/**
 * @brief All other cases for tuple finding
 *
 * In all other cases, we continue to iterate through recursively until the right tuple
 * is found. Note that the lambda must have the right type casting when calling it,
 * as it also needs to receive itself.
 *
 * @tparam LambdaType Inferred type of the lambda
 * @tparam I Inferred index of the iteration
 * @tparam T Inferred types inside the tuple
 * @param t Tuple being searched
 * @param idx Index de1sired
 * @param lambda Lambda function to run
 */
template<typename LambdaType, std::size_t I = 0, typename... T>
  inline typename std::enable_if < I<sizeof...(T), void>::type find_tuple(const std::tuple<T...>& t,
                                                                          std::size_t idx,
                                                                          const LambdaType& lambda)
{
    // if we've found the index, we can execute the lambda and return out
    if (I == idx)
    {
        auto& elem = std::get<I>(t);
        lambda(elem);
    }
    // otherwise we continue to recurse through until we find the lambda
    else
    {
        find_tuple<LambdaType, I + 1, T...>(t, idx, lambda);
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

/**
 * @brief Set the device to use
 *
 * @param argc Number of program arguments
 * @param argv Program arguments
 * @return DeviceSelector Device that should be used for the framework
 */
inline DeviceSelector set_device(int argc, char* argv[])
{
    DeviceSelector device = DeviceSelector::AUTO;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--device=") == 0)
        {
            arg           = arg.substr(9);
            const char* s = &arg.c_str()[0];
            if (strcmp(s, "device") == 0)
            {
                device = DeviceSelector::DEVICE;
            }
            else if (strcmp(s, "host") == 0)
            {
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

/**
 * @brief Set the size of the data N
 *
 * @param argc Number of program arguments
 * @param argv Program arguments
 * @return int Size to be used for the data
 */
inline int set_N(int argc, char* argv[])
{
    int N = 5;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--N=") == 0)
        {
            N = std::atoi(arg.substr(4).c_str());
            break;
        }
    }
    std::cout << "N = " << N << std::endl;
    return N;
}

/**
 * @brief Set if reordering should be used
 *
 * @param argc Number of program arguments
 * @param argv Program arguments
 * @return bool Desired reordering option
 */
inline bool set_reordering(int argc, char* argv[])
{
    bool flag = false;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--reordering") == 0)
        {
            flag = true;
            break;
        }
    }
    std::string s(flag ? "true" : "false");
    std::cout << "reordering = " << s << std::endl;
    return flag;
}

/**
 * @brief Set if initialization should be done
 *
 * @param argc Number of program arguments
 * @param argv Program arguments
 * @return int Desired initalization option
 */
inline int set_initialize(int argc, char* argv[])
{
    bool flag = true;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--no-initialize") == 0)
        {
            flag = false;
            break;
        }
    }
    std::string s(flag ? "true" : "false");
    std::cout << "initialize = " << s << std::endl;
    return flag;
}

/**
 * @brief Set the number of optimization simulations to run
 *
 * @param argc Number of program arguments
 * @param argv Program arguments
 * @return int Desired number of simulations
 */
inline int set_num_sims(int argc, char* argv[])
{
    int N = 5;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--num_sims=") == 0)
        {
            N = std::atoi(arg.substr(11).c_str());
            break;
        }
    }
    std::cout << "num_sims = " << N << std::endl;
    return N;
}

/**
 * @brief Set the number of times chains should be run for profiling
 *
 * @param argc Number of program arguments
 * @param argv Program arguments
 * @return int Desired number of chain runs
 */
inline int set_num_chain_runs(int argc, char* argv[])
{
    int N = 1;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--chain_runs=") == 0)
        {
            N = std::atoi(arg.substr(13).c_str());
            break;
        }
    }
    std::cout << "chain_runs = " << N << std::endl;
    return N;
}

/**
 * @brief Set the number of output rows to print
 *
 * @param argc Number of program arguments
 * @param argv Program arguments
 * @return int Desired number of output rows
 */
inline int set_num_output_truncate(int argc, char* argv[])
{
    int N = 25;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--output_truncate=") == 0)
        {
            N = std::atoi(arg.substr(18).c_str());
            break;
        }
    }
    std::cout << "output_truncate = " << N << std::endl;
    return N;
}

/**
 * @brief Set the save prefix for results, includes folder
 *
 * @param argc Number of program arguments
 * @param argv Program arguments
 * @return std::string The save prefix
 */
inline std::string set_save_prefix(int argc, char* argv[])
{
    std::string prefix = "output";
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--save_prefix=") == 0)
        {
            prefix = arg.substr(14);
            break;
        }
    }
    std::cout << "save_prefix = " << prefix << std::endl;

    // then also get the current date and time
    auto t         = std::time(nullptr);
    auto curr_time = *std::localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(&curr_time, "%d-%m-%Y_%H-%M-%S");
    return prefix + "_" + oss.str() + "_";
}

#ifdef DYNTUNE_SINGLE_CHAIN_RUN
/**
 * @brief Set the ID for the single chain that should be run, useful for debugging
 *
 * @param argc Number of program arguments
 * @param argv Program arguments
 * @return int ID of the chain to run
 */
inline unsigned int set_single_chain_run(int argc, char* argv[])
{
    unsigned int N = 0;
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg.find("--single_chain=") == 0)
        {
            N = std::atoi(arg.substr(15).c_str());
            break;
        }
    }
    std::cout << "single_chain = " << N << std::endl;
    return N;
}
#endif


/**
 * @brief Helper function to print if something is a reference
 *
 * @tparam T Inferred type
 * @param arg The argument to the function
 */
template<typename T>
inline void print_is_reference(const T& arg)
{
    if constexpr (std::is_reference_v<T>)
    {
        std::cout << "The argument is a reference." << std::endl;
    }
    else
    {
        std::cout << "The argument is not a reference." << std::endl;
    }
}
