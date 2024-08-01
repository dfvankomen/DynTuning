#pragma once

#include "view.hpp"

#include <string>
#include <tuple>

/**
 * @brief Create a compile-time hash from a string
 *
 * We need the name string to be embedded in the template args, but std::string
 * is not structural and does not use external linkage, so it is not allowed.
 * So, we store the hash of the string instead, and this provides the compile-time hash.
 * Credit: https://www.reddit.com/r/cpp/comments/jkw84k/strings_in_switch_statements_using_constexp/
 *
 * @param str String to hash
 * @return constexpr std::size_t Hashed string
 */
constexpr std::size_t hash(const char* str)
{
    const long long p            = 131;
    const long long m            = 4294967291; // 2^32 - 5, largest 32 bit prime
    long long total              = 0;
    long long current_multiplier = 1;
    for (int i = 0; str[i] != '\0'; ++i)
    {
        total              = (total + current_multiplier * str[i]) % m;
        current_multiplier = (current_multiplier * p) % m;
    }
    return total;
}

/**
 * @brief Struct to store the Hashed Name
 *
 * Provides a struct to hold the hash as part of the template parameters.
 * Intends to create a unique type for each "name" string, though collisions could occur.
 *
 * @tparam NameHash
 */
template<std::size_t NameHash>
struct HashedName
{
    constexpr HashedName() {};
    static constexpr std::size_t hash = NameHash;
};


/**
 * @brief Finds value of tuple based on hash, fallback function
 *
 * Fall back if I == size of tuple, returns -1
 *
 * @tparam HashToFind Input Hash to Find
 * @tparam I Inferred index type
 * @tparam Tp Internal tuple type
 * @param t Tuple to search through
 * @return constexpr std::enable_if<I == sizeof...(Tp), std::size_t>::type
 */
template<std::size_t HashToFind, std::size_t I = 0, typename... Tp>
constexpr typename std::enable_if<I == sizeof...(Tp), std::size_t>::type find(
  const std::tuple<Tp...>& t)
{
    return -1;
}

/**
 * @brief Find index of tuple that matches hash
 *
 * Returns the index of in the tuple for the name that matches HashToFind
 *
 * @tparam HashToFind Input Hash to find
 * @tparam I Inferred index type
 * @tparam Tp Internal tuple type
 * @param t Tuple to search through
 */
template<std::size_t HashToFind, std::size_t I = 0, typename... Tp>
  constexpr typename std::enable_if <
  I<sizeof...(Tp), std::size_t>::type find(const std::tuple<Tp...>& t)
{
    if (std::get<I>(t).hash == HashToFind)
    {
        return I;
    }
    else
    {
        return find<HashToFind, I + 1, Tp...>(t);
    }
}

/*
// I cannot seem to get this to work
// Compiler complains that "first" cannot be used as a constant
// Main function to create the data_names tuple
template <typename First>
inline constexpr auto make_data_names(const First first) {
    return std::make_tuple(first);
}
template <typename First, typename... Rest>
inline constexpr auto make_data_names(const First first, const Rest... rest) {
    return std::tuple_cat(std::make_tuple(HashedName<hash(first)>()), make_data_names(rest...));
}
*/



/*
template <std::size_t N, typename NamesType, typename... ParameterTypes>
class DataManager
{
  using DataTuplesType = get_tuple_type<ParameterTypes...>::type;

  public:
    //DataManager(std::vector<std::string> names, std::tuple<ParameterTypes&...> params)
    DataManager(NamesType names, std::tuple<ParameterTypes&...> params)
        : data_(create_views(params))
    {};

    DataTuplesType data_;
    NamesType names_;
};

template <typename NamesType, typename... ParameterTypes>
inline auto MakeDataManager(NamesType names, std::tuple<ParameterTypes&...> params)
{
    constexpr std::size_t N = std::tuple_size_v<decltype(params)>;
    return DataManager<N, NamesType, ParameterTypes...>(names, params);
}
*/

// template <typename DataManagerType>
// inline auto get_data(const char* s, DataManagerType& DataManager) {
//     return std::get<find<hash(s)>(DataManager.names_)>(DataManager.data_);
// }
