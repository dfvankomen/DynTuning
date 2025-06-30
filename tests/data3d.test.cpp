
#include "Kokkos_Core.hpp"
#include "Kokkos_Random.hpp"
#include "common.hpp"
#include "data.hpp"
#include "data_transfers.hpp"
#include "range.hpp"
#include "view.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;
typedef Kokkos::MDRangePolicy<DeviceExecSpace, Kokkos::Rank<3>> device_rank3_range_policy;


#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))


TEST_CASE("Eigen Tensor to View Tests", "[views-3d]")
{
    const size_t N = 2;
    const size_t M = 3;
    const size_t K = 4;

    {
        DynMatrix3D A(N, M, K);
        DynMatrix3D B(N, M, K);

        // initialize the matrix
        int ijk = 1;
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < M; j++)
            {
                for (size_t k = 0; k < K; k++)
                {
                    A(i, j, k) = static_cast<double>(ijk);
                    B(i, j, k) = static_cast<double>(ijk);
                    ijk++;
                }
            }
        }

        constexpr auto data_names =
          std::make_tuple(HashedName<hash("A")>(), HashedName<hash("B")>());

        auto data_views = create_views(pack(A, B));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);

        // and get an easy access to the views
        auto& A_host = std::get<0>(A_views);
        auto& B_host = std::get<0>(B_views);

        // verify that the view indices match what we're expecting above
        REQUIRE_THAT(A_host(0, 0, 0), Catch::Matchers::WithinULP(1.0, 0));
        REQUIRE_THAT(A_host(0, 0, 1), Catch::Matchers::WithinULP(2.0, 0));
        REQUIRE_THAT(A_host(0, 0, 2), Catch::Matchers::WithinULP(3.0, 0));
        REQUIRE_THAT(A_host(0, 0, 3), Catch::Matchers::WithinULP(4.0, 0));
        // test the last row in the last "sheet"
        REQUIRE_THAT(A_host(1, 2, 0), Catch::Matchers::WithinULP(21.0, 0));
        REQUIRE_THAT(A_host(1, 2, 1), Catch::Matchers::WithinULP(22.0, 0));
        REQUIRE_THAT(A_host(1, 2, 2), Catch::Matchers::WithinULP(23.0, 0));
        REQUIRE_THAT(A_host(1, 2, 3), Catch::Matchers::WithinULP(24.0, 0));

        // verify that the view indices match what we're expecting above
        REQUIRE_THAT(B_host(0, 0, 0), Catch::Matchers::WithinULP(1.0, 0));
        REQUIRE_THAT(B_host(0, 0, 1), Catch::Matchers::WithinULP(2.0, 0));
        REQUIRE_THAT(B_host(0, 0, 2), Catch::Matchers::WithinULP(3.0, 0));
        REQUIRE_THAT(B_host(0, 0, 3), Catch::Matchers::WithinULP(4.0, 0));
        // test the last row in the last "sheet"
        REQUIRE_THAT(B_host(1, 2, 0), Catch::Matchers::WithinULP(21.0, 0));
        REQUIRE_THAT(B_host(1, 2, 1), Catch::Matchers::WithinULP(22.0, 0));
        REQUIRE_THAT(B_host(1, 2, 2), Catch::Matchers::WithinULP(23.0, 0));
        REQUIRE_THAT(B_host(1, 2, 3), Catch::Matchers::WithinULP(24.0, 0));
    }
}

TEST_CASE("Rank 3 View Data Transfer Tests", "[transfer][views-3d]")
{

    const size_t N = 2;
    const size_t M = 3;
    const size_t K = 4;

    {
        DynMatrix3D A(N, M, K);
        DynMatrix3D B(N, M, K);

        // initialize the matrix
        int ijk = 1;
        for (size_t i = 0; i < N; i++)
        {
            for (size_t j = 0; j < M; j++)
            {
                for (size_t k = 0; k < K; k++)
                {
                    A(i, j, k) = static_cast<double>(ijk);
                    B(i, j, k) = static_cast<double>(ijk);
                    ijk++;
                }
            }
        }

        constexpr auto data_names =
          std::make_tuple(HashedName<hash("A")>(), HashedName<hash("B")>());

        auto data_views = create_views(pack(A, B));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);

        auto views_dimension_transposed = repack_views(data_views);

        // TODO: replace with proper templated version
        // and get an easy access to the views
        auto& A_host = std::get<0>(A_views);
        auto& B_host = std::get<0>(B_views);


        // transfer the data
        // Since we can't easily access the data on the device, we have to do the transfer, do a
        // modification, and then check upon return if the values are correct
        for (size_t i_data = 0; i_data < 2; i_data++)
            transfer_data_host_to_device(i_data, views_dimension_transposed);

        // micro kokkos kernel just to see if the data actually transferred by doing some simple
        // math
        auto& A_device = std::get<1>(A_views);
        auto& B_device = std::get<1>(B_views);
        Kokkos::parallel_for("Update a and b Loop",
                             device_rank3_range_policy({ 0, 0, 0 }, { N, M, K }),
                             KOKKOS_LAMBDA(const int i, const int j, const int k) {
                                 // simple calculations, just to make sure things are working right
                                 A_device(i, j, k) = 500 + A_device(i, j, k);
                                 B_device(i, j, k) = 10.0 * B_device(i, j, k);
                             });

        // TRANSFER BACK TO HOST
        for (size_t i_data = 0; i_data < 2; i_data++)
            transfer_data_device_to_host(i_data, views_dimension_transposed);

        // now we do some comparisons on the host data to make sure they're what we expect.
        // A should contain value from 501 - 521 now
        // B should contain values 10, 20, 30, 40, etc. now
        REQUIRE_THAT(A_host(0, 0, 0), Catch::Matchers::WithinRel(501, 0.00001));
        REQUIRE_THAT(A_host(0, 0, 1), Catch::Matchers::WithinRel(502, 0.00001));
        REQUIRE_THAT(A_host(0, 0, 2), Catch::Matchers::WithinRel(503, 0.00001));
        REQUIRE_THAT(A_host(0, 0, 3), Catch::Matchers::WithinRel(504, 0.00001));
        // last sheet last row
        REQUIRE_THAT(A_host(1, 2, 0), Catch::Matchers::WithinRel(521, 0.00001));
        REQUIRE_THAT(A_host(1, 2, 1), Catch::Matchers::WithinRel(522, 0.00001));
        REQUIRE_THAT(A_host(1, 2, 2), Catch::Matchers::WithinRel(523, 0.00001));
        REQUIRE_THAT(A_host(1, 2, 3), Catch::Matchers::WithinRel(524, 0.00001));

        REQUIRE_THAT(B_host(0, 0, 0), Catch::Matchers::WithinRel(10, 0.00001));
        REQUIRE_THAT(B_host(0, 0, 1), Catch::Matchers::WithinRel(20, 0.00001));
        REQUIRE_THAT(B_host(0, 0, 2), Catch::Matchers::WithinRel(30, 0.00001));
        REQUIRE_THAT(B_host(0, 0, 3), Catch::Matchers::WithinRel(40, 0.00001));
        // last sheet last row
        REQUIRE_THAT(B_host(1, 2, 0), Catch::Matchers::WithinRel(210, 0.00001));
        REQUIRE_THAT(B_host(1, 2, 1), Catch::Matchers::WithinRel(220, 0.00001));
        REQUIRE_THAT(B_host(1, 2, 2), Catch::Matchers::WithinRel(230, 0.00001));
        REQUIRE_THAT(B_host(1, 2, 3), Catch::Matchers::WithinRel(240, 0.00001));
    }
}


TEST_CASE("Test Multiple View Construction (with 3D)", "[views][views-3d]")
{

    // size of the input vector
    const size_t N    = 100;
    const size_t ncol = 5;
    const size_t nrow = 10;

    const size_t N_t = 2;
    const size_t M_t = 3;
    const size_t K_t = 4;
    {
        std::vector<double> a(N);
        std::vector<double> b(N);
        DynMatrix2D X(nrow, ncol);
        DynMatrix2D Y(nrow, ncol);
        DynMatrix3D alpha(N_t, M_t, K_t);
        DynMatrix3D beta(N_t, M_t, K_t);

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
        int ijk = 1;
        for (size_t i = 0; i < N_t; i++)
        {
            for (size_t j = 0; j < M_t; j++)
            {
                for (size_t k = 0; k < K_t; k++)
                {
                    alpha(i, j, k) = static_cast<double>(ijk);
                    beta(i, j, k)  = static_cast<double>(ijk);
                    ijk++;
                }
            }
        }

        constexpr auto data_names = std::make_tuple(HashedName<hash("a")>(),
                                                    HashedName<hash("b")>(),
                                                    HashedName<hash("X")>(),
                                                    HashedName<hash("Y")>(),
                                                    HashedName<hash("alpha")>(),
                                                    HashedName<hash("beta")>());

        auto data_views = create_views(pack(a, b, X, Y, alpha, beta));

        auto& a_views     = std::get<find<hash("a")>(data_names)>(data_views);
        auto& b_views     = std::get<find<hash("b")>(data_names)>(data_views);
        auto& X_views     = std::get<find<hash("X")>(data_names)>(data_views);
        auto& Y_views     = std::get<find<hash("Y")>(data_names)>(data_views);
        auto& alpha_views = std::get<find<hash("alpha")>(data_names)>(data_views);
        auto& beta_views  = std::get<find<hash("beta")>(data_names)>(data_views);

        auto& a_host     = std::get<0>(a_views);
        auto& b_host     = std::get<0>(b_views);
        auto& X_host     = std::get<0>(X_views);
        auto& Y_host     = std::get<0>(Y_views);
        auto& alpha_host = std::get<0>(alpha_views);
        auto& beta_host  = std::get<0>(beta_views);


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

        // and then the tensors
        // verify that the view indices match what we're expecting above
        REQUIRE_THAT(alpha_host(0, 0, 0), Catch::Matchers::WithinULP(1.0, 0));
        REQUIRE_THAT(alpha_host(0, 0, 1), Catch::Matchers::WithinULP(2.0, 0));
        REQUIRE_THAT(alpha_host(0, 0, 2), Catch::Matchers::WithinULP(3.0, 0));
        REQUIRE_THAT(alpha_host(0, 0, 3), Catch::Matchers::WithinULP(4.0, 0));
        // test the last row in the last "sheet"
        REQUIRE_THAT(alpha_host(1, 2, 0), Catch::Matchers::WithinULP(21.0, 0));
        REQUIRE_THAT(alpha_host(1, 2, 1), Catch::Matchers::WithinULP(22.0, 0));
        REQUIRE_THAT(alpha_host(1, 2, 2), Catch::Matchers::WithinULP(23.0, 0));
        REQUIRE_THAT(alpha_host(1, 2, 3), Catch::Matchers::WithinULP(24.0, 0));

        // verify that the view indices match what we're expecting above
        REQUIRE_THAT(beta_host(0, 0, 0), Catch::Matchers::WithinULP(1.0, 0));
        REQUIRE_THAT(beta_host(0, 0, 1), Catch::Matchers::WithinULP(2.0, 0));
        REQUIRE_THAT(beta_host(0, 0, 2), Catch::Matchers::WithinULP(3.0, 0));
        REQUIRE_THAT(beta_host(0, 0, 3), Catch::Matchers::WithinULP(4.0, 0));
        // test the last row in the last "sheet"
        REQUIRE_THAT(beta_host(1, 2, 0), Catch::Matchers::WithinULP(21.0, 0));
        REQUIRE_THAT(beta_host(1, 2, 1), Catch::Matchers::WithinULP(22.0, 0));
        REQUIRE_THAT(beta_host(1, 2, 2), Catch::Matchers::WithinULP(23.0, 0));
        REQUIRE_THAT(beta_host(1, 2, 3), Catch::Matchers::WithinULP(24.0, 0));
    }
}
