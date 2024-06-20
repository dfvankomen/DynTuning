#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"
#include "data_transfers.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <utility>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;
typedef Kokkos::MDRangePolicy<DeviceExecSpace, Kokkos::Rank<2>> device_rank2_range_policy;


#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))

// inner most loop calls the get_v multiple times
// template<typename... T, std::size_t... I>
// inline static auto transpose_tuple_inner(std::tuple<T...> t, size_t& j,
// std::integer_sequence<std::size_t, I...>) {
//     return get_v(I, j, t);
// }

// template<typename... T>
// inline static auto transpose_tuple_outer(std::tuple<T...> t, const size_t& j)
// {
//     return std::make_tuple(transpose_tuple_inner(std::make_index_sequence<sizeof...(T)> {}, j,
//     t)...);
// }

// template<typename... T>
// inline static auto transpose_tuple(std::tuple<T...> t)
// {
//     return std::make_tuple(transpose_tuple_outer(t, 0), transpose_tuple_outer(t, 1),
//     transpose_tuple_outer(t, 2));
// }

TEST_CASE("Vector to View Tests", "views")
{

    // size of the input vector
    const size_t N = 100;
    Kokkos::initialize();
    {
        std::vector<double> a(N);
        std::iota(a.begin(), a.end(), 1.0);
        std::vector<double> b(N);
        std::iota(b.begin(), b.end(), 1.0);

        constexpr auto data_names =
          std::make_tuple(HashedName<hash("a")>(), HashedName<hash("b")>());

        auto data_views = create_views(pack(a, b));

        auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);
        auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);


        SECTION("Testing View Tuple Reorder")
        {
            auto views_dimension_flopped =
              std::make_tuple(std::make_tuple(get_v(0, 0, data_views), get_v(1, 0, data_views)),
                              std::make_tuple(get_v(0, 1, data_views), get_v(1, 1, data_views)),
                              std::make_tuple(get_v(0, 2, data_views), get_v(1, 2, data_views)));
            // auto views_flopped_new = transpose_tuple(data_views);

            // after flopping the dimensions, the views should now be in order of [host_views,
            // device_views, scratch_views]
        }

        REQUIRE(N == 100);
    }
    Kokkos::finalize();
}

TEST_CASE("Rank 1 View Data Transfer Tests", "data-transfer")
{
    // size of the input vector
    const size_t N = 100;
    Kokkos::initialize();
    {
        std::vector<double> a(N);
        std::vector<double> b(N);
        std::iota(a.begin(), a.end(), 1.0);
        std::iota(b.begin(), b.end(), 1.0);

        constexpr auto data_names =
          std::make_tuple(HashedName<hash("a")>(), HashedName<hash("b")>());

        auto data_views = create_views(pack(a, b));

        auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);
        auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);

        // TODO: replace with proper templated version
        auto views_dimension_flopped =
          std::make_tuple(std::make_tuple(get_v(0, 0, data_views), get_v(1, 0, data_views)),
                          std::make_tuple(get_v(0, 1, data_views), get_v(1, 1, data_views)),
                          std::make_tuple(get_v(0, 2, data_views), get_v(1, 2, data_views)));

        // and get an easy access to the views
        auto& a_host = std::get<0>(a_views);
        auto& b_host = std::get<0>(b_views);

        // TRANSFER THE DATA
        // Since we can't easily access the data on the device, we have to do the transfer, do a
        // modification, and then check upon return if the values are correct
        for (size_t i_data = 0; i_data < 2; i_data++)
        {
            transfer_data_host_to_device(i_data, views_dimension_flopped);
        }

        // micro kokkos kernel just to see if the data actually transferred by doing some simple
        // math
        auto& a_device = std::get<1>(a_views);
        auto& b_device = std::get<1>(b_views);
        Kokkos::parallel_for("A Update Loop",
                             device_range_policy(0, N),
                             KOKKOS_LAMBDA(const int i) {
                                 a_device(i) = 500 + a_device(i);
                                 b_device(i) = 10.0 * b_device(i);
                             });

        // TRANSFER BACK TO HOST
        for (size_t i_data = 0; i_data < 2; i_data++)
        {
            transfer_data_device_to_host(i_data, views_dimension_flopped);
        }


        // now we do some comparisons on the host data to make sure they're what we expect.
        // A should contain value from 501 - 601 now
        // B should contain values 10, 20, 30, 40, etc. now
        REQUIRE_THAT(a_host(0), Catch::Matchers::WithinRel(501, 0.00001));
        REQUIRE_THAT(a_host(1), Catch::Matchers::WithinRel(502, 0.00001));
        REQUIRE_THAT(a_host(2), Catch::Matchers::WithinRel(503, 0.00001));
        REQUIRE_THAT(a_host(3), Catch::Matchers::WithinRel(504, 0.00001));
        REQUIRE_THAT(a_host(4), Catch::Matchers::WithinRel(505, 0.00001));

        REQUIRE_THAT(b_host(0), Catch::Matchers::WithinRel(10, 0.00001));
        REQUIRE_THAT(b_host(1), Catch::Matchers::WithinRel(20, 0.00001));
        REQUIRE_THAT(b_host(2), Catch::Matchers::WithinRel(30, 0.00001));
        REQUIRE_THAT(b_host(3), Catch::Matchers::WithinRel(40, 0.00001));
        REQUIRE_THAT(b_host(4), Catch::Matchers::WithinRel(50, 0.00001));

        // data transfer should be *exact* since we're using the same data types, so WithinULP will
        // verify consistency NOTE: on some arch, it's *possible* (but unlikely) that there will be
        // minor differences A tests
        REQUIRE_THAT(a_host(0), Catch::Matchers::WithinULP(501.0, 0));
        REQUIRE_THAT(a_host(1), Catch::Matchers::WithinULP(502.0, 0));
        REQUIRE_THAT(a_host(2), Catch::Matchers::WithinULP(503.0, 0));
        REQUIRE_THAT(a_host(3), Catch::Matchers::WithinULP(504.0, 0));
        REQUIRE_THAT(a_host(4), Catch::Matchers::WithinULP(505.0, 0));

        // B tests
        REQUIRE_THAT(b_host(0), Catch::Matchers::WithinULP(10.0, 0));
        REQUIRE_THAT(b_host(1), Catch::Matchers::WithinULP(20.0, 0));
        REQUIRE_THAT(b_host(2), Catch::Matchers::WithinULP(30.0, 0));
        REQUIRE_THAT(b_host(3), Catch::Matchers::WithinULP(40.0, 0));
        REQUIRE_THAT(b_host(4), Catch::Matchers::WithinULP(50.0, 0));
    }
    Kokkos::finalize();
}


TEST_CASE("Rank 2 View Data Transfer Tests", "data-transfer")
{
    // size of the input vector
    const size_t N = 3; // num rows
    const size_t M = 5; // num cols
    Kokkos::initialize();
    {
        // first value in gives .rows() and second gives .cols()
        // eigen stores data in **column major order**.
        Eigen::MatrixXd a(N, M);
        Eigen::MatrixXd b(N, M);

        // initialize both matrices to be identical
        int ij = 1;
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < M; j++)
            {
                a(i, j) = static_cast<double>(ij);
                b(i, j) = static_cast<double>(ij);
                ij++;
            }
        }
        // a and b are matrices from 1 to N * M now
        // print the matrix

        constexpr auto data_names =
          std::make_tuple(HashedName<hash("a")>(), HashedName<hash("b")>());

        auto data_views = create_views(pack(a, b));

        auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);
        auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);

        // TODO: replace with proper templated version
        auto views_dimension_flopped =
          std::make_tuple(std::make_tuple(get_v(0, 0, data_views), get_v(1, 0, data_views)),
                          std::make_tuple(get_v(0, 1, data_views), get_v(1, 1, data_views)),
                          std::make_tuple(get_v(0, 2, data_views), get_v(1, 2, data_views)));

        // and get an easy access to the views
        auto& a_host = std::get<0>(a_views);
        auto& b_host = std::get<0>(b_views);

        // verify that the view indices match what we're expecting above
        REQUIRE_THAT(a_host(0, 0), Catch::Matchers::WithinRel(1, 0.00001));
        REQUIRE_THAT(a_host(0, 1), Catch::Matchers::WithinRel(2, 0.00001));
        REQUIRE_THAT(a_host(0, 2), Catch::Matchers::WithinRel(3, 0.00001));
        REQUIRE_THAT(a_host(0, 3), Catch::Matchers::WithinRel(4, 0.00001));
        REQUIRE_THAT(a_host(0, 4), Catch::Matchers::WithinRel(5, 0.00001));

        // TRANSFER THE DATA
        // Since we can't easily access the data on the device, we have to do the transfer, do a
        // modification, and then check upon return if the values are correct
        for (size_t i_data = 0; i_data < 2; i_data++)
            transfer_data_host_to_device(i_data, views_dimension_flopped);

        // micro kokkos kernel just to see if the data actually transferred by doing some simple
        // math
        auto& a_device = std::get<1>(a_views);
        auto& b_device = std::get<1>(b_views);
        Kokkos::parallel_for("Update a and b Loop",
                             device_rank2_range_policy({ 0, 0 }, { N, M }),
                             KOKKOS_LAMBDA(const int i, const int j) {
                                 // simple calculations, just to make sure things are working right
                                 a_device(i, j) = 500 + a_device(i, j);
                                 b_device(i, j) = 10.0 * b_device(i, j);
                             });

        // TRANSFER BACK TO HOST
        for (size_t i_data = 0; i_data < 2; i_data++)
            transfer_data_device_to_host(i_data, views_dimension_flopped);

        // now we do some comparisons on the host data to make sure they're what we expect.
        // A should contain value from 501 - 601 now
        // B should contain values 10, 20, 30, 40, etc. now
        REQUIRE_THAT(a_host(0, 0), Catch::Matchers::WithinRel(501, 0.00001));
        REQUIRE_THAT(a_host(0, 1), Catch::Matchers::WithinRel(502, 0.00001));
        REQUIRE_THAT(a_host(0, 2), Catch::Matchers::WithinRel(503, 0.00001));
        REQUIRE_THAT(a_host(0, 3), Catch::Matchers::WithinRel(504, 0.00001));
        REQUIRE_THAT(a_host(0, 4), Catch::Matchers::WithinRel(505, 0.00001));

        REQUIRE_THAT(b_host(0, 0), Catch::Matchers::WithinRel(10, 0.00001));
        REQUIRE_THAT(b_host(0, 1), Catch::Matchers::WithinRel(20, 0.00001));
        REQUIRE_THAT(b_host(0, 2), Catch::Matchers::WithinRel(30, 0.00001));
        REQUIRE_THAT(b_host(0, 3), Catch::Matchers::WithinRel(40, 0.00001));
        REQUIRE_THAT(b_host(0, 4), Catch::Matchers::WithinRel(50, 0.00001));

        // data transfer should be *exact* since we're using the same data types, so WithinULP will
        // verify consistency NOTE: on some arch, it's *possible* (but unlikely) that there will be
        // minor differences A tests
        REQUIRE_THAT(a_host(0, 0), Catch::Matchers::WithinULP(501.0, 0));
        REQUIRE_THAT(a_host(0, 1), Catch::Matchers::WithinULP(502.0, 0));
        REQUIRE_THAT(a_host(0, 2), Catch::Matchers::WithinULP(503.0, 0));
        REQUIRE_THAT(a_host(0, 3), Catch::Matchers::WithinULP(504.0, 0));
        REQUIRE_THAT(a_host(0, 4), Catch::Matchers::WithinULP(505.0, 0));

        // B tests
        REQUIRE_THAT(b_host(0, 0), Catch::Matchers::WithinULP(10.0, 0));
        REQUIRE_THAT(b_host(0, 1), Catch::Matchers::WithinULP(20.0, 0));
        REQUIRE_THAT(b_host(0, 2), Catch::Matchers::WithinULP(30.0, 0));
        REQUIRE_THAT(b_host(0, 3), Catch::Matchers::WithinULP(40.0, 0));
        REQUIRE_THAT(b_host(0, 4), Catch::Matchers::WithinULP(50.0, 0));
    }
    Kokkos::finalize();
}