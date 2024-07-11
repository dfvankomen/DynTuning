#include "data.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"
#include "common.hpp"
#include "data_transfers.hpp"
#include "range.hpp"
#include "view.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;
typedef Kokkos::MDRangePolicy<DeviceExecSpace, Kokkos::Rank<2>> device_rank2_range_policy;


#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))


TEST_CASE("Vector to View Tests", "views")
{

    // size of the input vector
    const size_t N = 100;
    Kokkos::initialize();
    {
        std::vector<double> a(N);
        std::vector<double> b(N);

        // data initilization
        std::iota(a.begin(), a.end(), 1.0);
        std::iota(b.begin(), b.end(), 1.0);

        constexpr auto data_names =
          std::make_tuple(HashedName<hash("a")>(), HashedName<hash("b")>());

        auto data_views = create_views(pack(a, b));

        auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);
        auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);

        auto& a_host = std::get<0>(a_views);
        auto& b_host = std::get<0>(b_views);

        // then check to make sure the values match up
        REQUIRE_THAT(a_host(0), Catch::Matchers::WithinULP(a[0], 0));
        REQUIRE_THAT(a_host(1), Catch::Matchers::WithinULP(a[1], 0));
        REQUIRE_THAT(a_host(2), Catch::Matchers::WithinULP(a[2], 0));
        REQUIRE_THAT(a_host(3), Catch::Matchers::WithinULP(a[3], 0));
        REQUIRE_THAT(a_host(4), Catch::Matchers::WithinULP(a[4], 0));
        REQUIRE_THAT(a_host(N - 5), Catch::Matchers::WithinULP(a[N - 5], 0));
        REQUIRE_THAT(a_host(N - 4), Catch::Matchers::WithinULP(a[N - 4], 0));
        REQUIRE_THAT(a_host(N - 3), Catch::Matchers::WithinULP(a[N - 3], 0));
        REQUIRE_THAT(a_host(N - 2), Catch::Matchers::WithinULP(a[N - 2], 0));
        REQUIRE_THAT(a_host(N - 1), Catch::Matchers::WithinULP(a[N - 1], 0));
        // check for b
        REQUIRE_THAT(b_host(0), Catch::Matchers::WithinULP(b[0], 0));
        REQUIRE_THAT(b_host(1), Catch::Matchers::WithinULP(b[1], 0));
        REQUIRE_THAT(b_host(2), Catch::Matchers::WithinULP(b[2], 0));
        REQUIRE_THAT(b_host(3), Catch::Matchers::WithinULP(b[3], 0));
        REQUIRE_THAT(b_host(4), Catch::Matchers::WithinULP(b[4], 0));
        REQUIRE_THAT(b_host(N - 5), Catch::Matchers::WithinULP(b[N - 5], 0));
        REQUIRE_THAT(b_host(N - 4), Catch::Matchers::WithinULP(b[N - 4], 0));
        REQUIRE_THAT(b_host(N - 3), Catch::Matchers::WithinULP(b[N - 3], 0));
        REQUIRE_THAT(b_host(N - 2), Catch::Matchers::WithinULP(b[N - 2], 0));
        REQUIRE_THAT(b_host(N - 1), Catch::Matchers::WithinULP(b[N - 1], 0));

        SECTION("Testing View Tuple Reorder")
        {
            auto views_dimension_transposed = repack_views(data_views);

            // after flopping the dimensions, the views should now be in order of [host_views,
            // device_views, scratch_views]
            // TODO: tests to make sure they line up
        }
    }
    Kokkos::finalize();
}

TEST_CASE("Eigen Matrix to View Tests", "views")
{

    // size of the input matrices
    const size_t N = 10;
    const size_t M = 5;
    Kokkos::initialize();
    {
        // first value in gives .rows() and second gives .cols()
        // eigen stores data in **column major order**.
        DynMatrix2D A(N, M);
        DynMatrix2D B(N, M);

        // initialize both matrices to be identical
        int ij = 1;
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < M; j++)
            {
                A(i, j) = static_cast<double>(ij);
                B(i, j) = static_cast<double>(ij);
                ij++;
            }
        }
        // a and b are matrices from 1 to N * M now

        constexpr auto data_names =
          std::make_tuple(HashedName<hash("A")>(), HashedName<hash("B")>());

        auto data_views = create_views(pack(A, B));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);

        // and get an easy access to the views
        auto& A_host = std::get<0>(A_views);
        auto& B_host = std::get<0>(B_views);

        // verify that the view indices match what we're expecting above
        REQUIRE_THAT(A_host(0, 0), Catch::Matchers::WithinULP(1.0, 0));
        REQUIRE_THAT(A_host(0, 1), Catch::Matchers::WithinULP(2.0, 0));
        REQUIRE_THAT(A_host(0, 2), Catch::Matchers::WithinULP(3.0, 0));
        REQUIRE_THAT(A_host(0, 3), Catch::Matchers::WithinULP(4.0, 0));
        REQUIRE_THAT(A_host(0, 4), Catch::Matchers::WithinULP(5.0, 0));
        // test the last row
        REQUIRE_THAT(A_host(9, 0), Catch::Matchers::WithinULP(46.0, 0));
        REQUIRE_THAT(A_host(9, 1), Catch::Matchers::WithinULP(47.0, 0));
        REQUIRE_THAT(A_host(9, 2), Catch::Matchers::WithinULP(48.0, 0));
        REQUIRE_THAT(A_host(9, 3), Catch::Matchers::WithinULP(49.0, 0));
        REQUIRE_THAT(A_host(9, 4), Catch::Matchers::WithinULP(50.0, 0));

        // verify that the view indices match what we're expecting above
        REQUIRE_THAT(B_host(0, 0), Catch::Matchers::WithinULP(1.0, 0));
        REQUIRE_THAT(B_host(0, 1), Catch::Matchers::WithinULP(2.0, 0));
        REQUIRE_THAT(B_host(0, 2), Catch::Matchers::WithinULP(3.0, 0));
        REQUIRE_THAT(B_host(0, 3), Catch::Matchers::WithinULP(4.0, 0));
        REQUIRE_THAT(B_host(0, 4), Catch::Matchers::WithinULP(5.0, 0));
        // test the last row
        REQUIRE_THAT(B_host(9, 0), Catch::Matchers::WithinULP(46.0, 0));
        REQUIRE_THAT(B_host(9, 1), Catch::Matchers::WithinULP(47.0, 0));
        REQUIRE_THAT(B_host(9, 2), Catch::Matchers::WithinULP(48.0, 0));
        REQUIRE_THAT(B_host(9, 3), Catch::Matchers::WithinULP(49.0, 0));
        REQUIRE_THAT(B_host(9, 4), Catch::Matchers::WithinULP(50.0, 0));
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
        auto views_dimension_transposed = repack_views(data_views);

        // and get an easy access to the views
        auto& a_host = std::get<0>(a_views);
        auto& b_host = std::get<0>(b_views);

        // TRANSFER THE DATA
        // Since we can't easily access the data on the device, we have to do the transfer, do a
        // modification, and then check upon return if the values are correct
        for (size_t i_data = 0; i_data < 2; i_data++)
        {
            transfer_data_host_to_device(i_data, views_dimension_transposed);
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
            transfer_data_device_to_host(i_data, views_dimension_transposed);
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
    const size_t N = 10; // num rows
    const size_t M = 5;  // num cols
    Kokkos::initialize();
    {
        // first value in gives .rows() and second gives .cols()
        // eigen stores data in **column major order**.
        DynMatrix2D A(N, M);
        DynMatrix2D B(N, M);

        // initialize both matrices to be identical
        int ij = 1;
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < M; j++)
            {
                A(i, j) = static_cast<double>(ij);
                B(i, j) = static_cast<double>(ij);
                ij++;
            }
        }
        // a and b are matrices from 1 to N * M now

        constexpr auto data_names =
          std::make_tuple(HashedName<hash("A")>(), HashedName<hash("B")>());

        auto data_views = create_views(pack(A, B));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);

        // TODO: replace with proper templated version
        auto views_dimension_transposed = repack_views(data_views);

        // and get an easy access to the views
        auto& A_host = std::get<0>(A_views);
        auto& B_host = std::get<0>(B_views);

        // TRANSFER THE DATA
        // Since we can't easily access the data on the device, we have to do the transfer, do a
        // modification, and then check upon return if the values are correct
        for (size_t i_data = 0; i_data < 2; i_data++)
            transfer_data_host_to_device(i_data, views_dimension_transposed);

        // micro kokkos kernel just to see if the data actually transferred by doing some simple
        // math
        auto& A_device = std::get<1>(A_views);
        auto& B_device = std::get<1>(B_views);
        Kokkos::parallel_for("Update a and b Loop",
                             device_rank2_range_policy({ 0, 0 }, { N, M }),
                             KOKKOS_LAMBDA(const int i, const int j) {
                                 // simple calculations, just to make sure things are working right
                                 A_device(i, j) = 500 + A_device(i, j);
                                 B_device(i, j) = 10.0 * B_device(i, j);
                             });

        // TRANSFER BACK TO HOST
        for (size_t i_data = 0; i_data < 2; i_data++)
            transfer_data_device_to_host(i_data, views_dimension_transposed);

        // now we do some comparisons on the host data to make sure they're what we expect.
        // A should contain value from 501 - 601 now
        // B should contain values 10, 20, 30, 40, etc. now
        REQUIRE_THAT(A_host(0, 0), Catch::Matchers::WithinRel(501, 0.00001));
        REQUIRE_THAT(A_host(0, 1), Catch::Matchers::WithinRel(502, 0.00001));
        REQUIRE_THAT(A_host(0, 2), Catch::Matchers::WithinRel(503, 0.00001));
        REQUIRE_THAT(A_host(0, 3), Catch::Matchers::WithinRel(504, 0.00001));
        REQUIRE_THAT(A_host(0, 4), Catch::Matchers::WithinRel(505, 0.00001));
        // test the last row
        REQUIRE_THAT(A_host(9, 0), Catch::Matchers::WithinRel(546, 0.00001));
        REQUIRE_THAT(A_host(9, 1), Catch::Matchers::WithinRel(547, 0.00001));
        REQUIRE_THAT(A_host(9, 2), Catch::Matchers::WithinRel(548, 0.00001));
        REQUIRE_THAT(A_host(9, 3), Catch::Matchers::WithinRel(549, 0.00001));
        REQUIRE_THAT(A_host(9, 4), Catch::Matchers::WithinRel(550, 0.00001));

        // B'e tests
        REQUIRE_THAT(B_host(0, 0), Catch::Matchers::WithinRel(10, 0.00001));
        REQUIRE_THAT(B_host(0, 1), Catch::Matchers::WithinRel(20, 0.00001));
        REQUIRE_THAT(B_host(0, 2), Catch::Matchers::WithinRel(30, 0.00001));
        REQUIRE_THAT(B_host(0, 3), Catch::Matchers::WithinRel(40, 0.00001));
        REQUIRE_THAT(B_host(0, 4), Catch::Matchers::WithinRel(50, 0.00001));
        // test the last row
        REQUIRE_THAT(B_host(9, 0), Catch::Matchers::WithinRel(460, 0.00001));
        REQUIRE_THAT(B_host(9, 1), Catch::Matchers::WithinRel(470, 0.00001));
        REQUIRE_THAT(B_host(9, 2), Catch::Matchers::WithinRel(480, 0.00001));
        REQUIRE_THAT(B_host(9, 3), Catch::Matchers::WithinRel(490, 0.00001));
        REQUIRE_THAT(B_host(9, 4), Catch::Matchers::WithinRel(500, 0.00001));

        // data transfer should be *exact* since we're using the same data types, so WithinULP will
        // verify consistency NOTE: on some arch, it's *possible* (but unlikely) that there will be
        // minor differences A tests
        REQUIRE_THAT(A_host(0, 0), Catch::Matchers::WithinULP(501.0, 0));
        REQUIRE_THAT(A_host(0, 1), Catch::Matchers::WithinULP(502.0, 0));
        REQUIRE_THAT(A_host(0, 2), Catch::Matchers::WithinULP(503.0, 0));
        REQUIRE_THAT(A_host(0, 3), Catch::Matchers::WithinULP(504.0, 0));
        REQUIRE_THAT(A_host(0, 4), Catch::Matchers::WithinULP(505.0, 0));
        // test the last row
        REQUIRE_THAT(A_host(9, 0), Catch::Matchers::WithinULP(546.0, 0));
        REQUIRE_THAT(A_host(9, 1), Catch::Matchers::WithinULP(547.0, 0));
        REQUIRE_THAT(A_host(9, 2), Catch::Matchers::WithinULP(548.0, 0));
        REQUIRE_THAT(A_host(9, 3), Catch::Matchers::WithinULP(549.0, 0));
        REQUIRE_THAT(A_host(9, 4), Catch::Matchers::WithinULP(550.0, 0));

        // B tests
        REQUIRE_THAT(B_host(0, 0), Catch::Matchers::WithinULP(10.0, 0));
        REQUIRE_THAT(B_host(0, 1), Catch::Matchers::WithinULP(20.0, 0));
        REQUIRE_THAT(B_host(0, 2), Catch::Matchers::WithinULP(30.0, 0));
        REQUIRE_THAT(B_host(0, 3), Catch::Matchers::WithinULP(40.0, 0));
        REQUIRE_THAT(B_host(0, 4), Catch::Matchers::WithinULP(50.0, 0));
        // test the last row
        REQUIRE_THAT(B_host(9, 0), Catch::Matchers::WithinULP(460.0, 0));
        REQUIRE_THAT(B_host(9, 1), Catch::Matchers::WithinULP(470.0, 0));
        REQUIRE_THAT(B_host(9, 2), Catch::Matchers::WithinULP(480.0, 0));
        REQUIRE_THAT(B_host(9, 3), Catch::Matchers::WithinULP(490.0, 0));
        REQUIRE_THAT(B_host(9, 4), Catch::Matchers::WithinULP(500.0, 0));
    }
    Kokkos::finalize();
}

TEST_CASE("Test Multiple View Construction", "views")
{

    // size of the input vector
    const size_t N    = 100;
    const size_t ncol = 5;
    const size_t nrow = 10;
    Kokkos::initialize();
    {
        std::vector<double> a(N);
        std::vector<double> b(N);
        DynMatrix2D X(nrow, ncol);
        DynMatrix2D Y(nrow, ncol);

        // data initialization
        std::iota(a.begin(), a.end(), 1.0);
        std::iota(b.begin(), b.end(), 1.0);
        int ij = 1;
        for (size_t i = 0; i < nrow; i++)
        {
            for (size_t j = 0; j < ncol; j++)
            {
                X(i, j) = static_cast<double>(ij);
                Y(i, j) = static_cast<double>(ij);
                ij++;
            }
        }
        // a and b are matrices from 1 to N * M now

        constexpr auto data_names = std::make_tuple(HashedName<hash("a")>(),
                                                    HashedName<hash("b")>(),
                                                    HashedName<hash("X")>(),
                                                    HashedName<hash("Y")>());

        auto data_views = create_views(pack(a, b, X, Y));

        auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);
        auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);
        auto& X_views = std::get<find<hash("X")>(data_names)>(data_views);
        auto& Y_views = std::get<find<hash("Y")>(data_names)>(data_views);

        auto& a_host = std::get<0>(a_views);
        auto& b_host = std::get<0>(b_views);
        auto& X_host = std::get<0>(X_views);
        auto& Y_host = std::get<0>(Y_views);


        // then check to make sure the values match up
        REQUIRE_THAT(a_host(0), Catch::Matchers::WithinULP(a[0], 0));
        REQUIRE_THAT(a_host(1), Catch::Matchers::WithinULP(a[1], 0));
        REQUIRE_THAT(a_host(N - 2), Catch::Matchers::WithinULP(a[N - 2], 0));
        REQUIRE_THAT(a_host(N - 1), Catch::Matchers::WithinULP(a[N - 1], 0));
        // check for b
        REQUIRE_THAT(b_host(0), Catch::Matchers::WithinULP(b[0], 0));
        REQUIRE_THAT(b_host(1), Catch::Matchers::WithinULP(b[1], 0));
        REQUIRE_THAT(b_host(N - 2), Catch::Matchers::WithinULP(b[N - 2], 0));
        REQUIRE_THAT(b_host(N - 1), Catch::Matchers::WithinULP(b[N - 1], 0));

        // verify the matrix views
        REQUIRE_THAT(X_host(0, 0), Catch::Matchers::WithinULP(1.0, 0));
        REQUIRE_THAT(X_host(0, 1), Catch::Matchers::WithinULP(2.0, 0));
        REQUIRE_THAT(X_host(9, 3), Catch::Matchers::WithinULP(49.0, 0));
        REQUIRE_THAT(X_host(9, 4), Catch::Matchers::WithinULP(50.0, 0));
        // check for y
        REQUIRE_THAT(Y_host(0, 0), Catch::Matchers::WithinULP(1.0, 0));
        REQUIRE_THAT(Y_host(0, 1), Catch::Matchers::WithinULP(2.0, 0));
        REQUIRE_THAT(Y_host(9, 3), Catch::Matchers::WithinULP(49.0, 0));
        REQUIRE_THAT(Y_host(9, 4), Catch::Matchers::WithinULP(50.0, 0));

        // // test the view tuple reorder
        // TODO: test this
    }
    Kokkos::finalize();
}

// template<typename T>
// std::string type_name();


TEST_CASE("Test creating views with Kokkos View datatype", "views")
{

    // size of the input vector
    const size_t N    = 10;
    const size_t ncol = 5;
    const size_t nrow = 10;

    Kokkos::initialize();
    {
        NDArrayView<double*> a("a", N);
        NDArrayView<double**> b("b", N, N);
        NDArrayView<double***> c("c", N, N, N);

        Kokkos::View<double**, Kokkos::KOKKOS_HOST> x("x", N, N);

        // assert at compile time that a, b, and c are going to work
        static_assert(IsKokkosView<decltype(a)>);
        static_assert(IsKokkosView<decltype(b)>);
        static_assert(IsKokkosView<decltype(c)>);

        using HostRangePolicy1D = typename RangePolicy<1, HostExecutionSpace>::type;
        using HostRangePolicy2D = typename RangePolicy<2, HostExecutionSpace>::type;
        using HostRangePolicy3D = typename RangePolicy<3, HostExecutionSpace>::type;

        Kokkos::parallel_for("Populate stuff", HostRangePolicy1D(0, N), KOKKOS_LAMBDA(const int i) {
            a(i) = i + 1;
        });
        Kokkos::parallel_for(
          "Populate stuff 2",
          HostRangePolicy2D({ 0, 0 }, { N, N }),
          KOKKOS_LAMBDA(const int i, const int j) { b(i, j) = (i + 1) * (j + 1); });
        Kokkos::parallel_for("Populate stuff 3",
                             HostRangePolicy3D({ 0, 0, 0 }, { N, N, N }),
                             KOKKOS_LAMBDA(const int i, const int j, const int k) {
                                 c(i, j, k) = (i + 1) * (j + 1) * (k + 1);
                             });

        Kokkos::Random_XorShift64_Pool<Kokkos::KOKKOS_HOST> rand_pool(0);
        // Kokkos::fill_random(x, rand_pool, 100.0);

        // actually create the views
        auto data_views           = create_views(pack(a, b, c));
        constexpr auto data_names = std::make_tuple(HashedName<hash("a")>(),
                                                    HashedName<hash("b")>(),
                                                    HashedName<hash("c")>());

        auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);
        auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);
        auto& c_views = std::get<find<hash("c")>(data_names)>(data_views);
        auto& a_host  = std::get<0>(a_views);
        auto& b_host  = std::get<0>(b_views);
        auto& c_host  = std::get<0>(c_views);

        REQUIRE_THAT(a_host(0), Catch::Matchers::WithinULP(a(0), 0));
        REQUIRE_THAT(a_host(1), Catch::Matchers::WithinULP(a(1), 0));
        REQUIRE_THAT(a_host(2), Catch::Matchers::WithinULP(a(2), 0));
        REQUIRE_THAT(a_host(3), Catch::Matchers::WithinULP(a(3), 0));

        // tests for b
        REQUIRE_THAT(b_host(0, 0), Catch::Matchers::WithinULP(b(0, 0), 0));
        REQUIRE_THAT(b_host(0, 1), Catch::Matchers::WithinULP(b(0, 1), 0));
        REQUIRE_THAT(b_host(0, 2), Catch::Matchers::WithinULP(b(0, 2), 0));
        REQUIRE_THAT(b_host(0, 3), Catch::Matchers::WithinULP(b(0, 3), 0));

        // tests for c
        REQUIRE_THAT(c_host(0, 0, 0), Catch::Matchers::WithinULP(c(0, 0, 0), 0));
        REQUIRE_THAT(c_host(0, 0, 1), Catch::Matchers::WithinULP(c(0, 0, 1), 0));
        REQUIRE_THAT(c_host(0, 0, 2), Catch::Matchers::WithinULP(c(0, 0, 2), 0));
        REQUIRE_THAT(c_host(0, 0, 3), Catch::Matchers::WithinULP(c(0, 0, 3), 0));
    }
    Kokkos::finalize();
}