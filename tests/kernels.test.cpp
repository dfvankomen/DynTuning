
#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"
#include "data_transfers.hpp"
#include "kernel_matvecmult.hpp"
#include "kernel_vectordot.hpp"
#include "view.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <utility>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;
typedef Kokkos::MDRangePolicy<DeviceExecSpace, Kokkos::Rank<2>> device_rank2_range_policy;


TEST_CASE("Kernel: Verify VectorDot Host and Device", "kernel")
{
    const size_t N = 100;

    Kokkos::initialize();
    {
        std::vector<double> a(N);
        std::vector<double> b(N);
        std::vector<double> c(N);
        std::vector<double> c_truth(N);

        // initialize a and b
        std::iota(a.begin(), a.end(), 1.0);

        // b can just be all 10's or something
        for (size_t i = 0; i < N; i++)
            b[i] = 10.0;

        for (size_t i = 0; i < N; i++)
            c_truth[i] = a[i] * b[i];

        constexpr auto data_names = std::make_tuple(HashedName<hash("a")>(),
                                                    HashedName<hash("b")>(),
                                                    HashedName<hash("c")>());

        auto data_views = create_views(pack(a, b, c));

        auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);
        auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);
        auto& c_views = std::get<find<hash("c")>(data_names)>(data_views);

        // used for the checks after we finish
        auto& c_host = std::get<0>(c_views);

        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        auto k = KernelVectorDot(options, std::as_const(a_views), std::as_const(b_views), c_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            REQUIRE_THAT(c_host(i), Catch::Matchers::WithinRel(c_truth[i], 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < N; i++)
            c[i] = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < N; i++)
            REQUIRE_THAT(c_host(i), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // verify that c_host is still zeros, to make sure kernel worked on device
        for (size_t i = 0; i < N; i++)
            REQUIRE_THAT(c_host(i), Catch::Matchers::WithinULP(0.0, 0));

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            REQUIRE_THAT(c_host(i), Catch::Matchers::WithinRel(c_truth[i], 1.e-10));

        // done!
    }
    Kokkos::finalize();
}


// TODO: reenable this test once parallel reduction is available!
#if 0
TEST_CASE("Kernel: Verify MatVecMult Host and Device", "kernel")
{
    const size_t N = 3;

    Kokkos::initialize();
    {
        DynMatrix2D A(N, N);
        std::vector<double> b(N);
        std::vector<double> c(N);
        std::vector<double> c_truth(N, 0.0);

        // initialize the A matrix
        int ij = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                A(i, j) = static_cast<double>(ij++);

        // b can just be all 10's
        for (size_t i = 0; i < N; i++)
            b[i] = 10.0;

        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                c_truth[i] += A(i, j) * b[j];


        constexpr auto data_names = std::make_tuple(HashedName<hash("A")>(),
                                                    HashedName<hash("b")>(),
                                                    HashedName<hash("c")>());

        auto data_views = create_views(pack(A, b, c));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);
        auto& c_views = std::get<find<hash("c")>(data_names)>(data_views);

        // used for the checks after we finish
        auto& A_host = std::get<0>(A_views);
        auto& c_host = std::get<0>(c_views);

        print_view(A_host);

        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        auto k = KernelMatVecMult(options, std::as_const(A_views), std::as_const(b_views), c_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            REQUIRE_THAT(c_host(i), Catch::Matchers::WithinRel(c_truth[i], 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < N; i++)
            c[i] = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < N; i++)
            REQUIRE_THAT(c_host(i), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);


        std::cout << "C truth: " << std::endl;
        for (size_t i = 0; i < N; i++)
            std::cout << c_truth[i] << " ";
        std::cout << std::endl;

        std::cout << "C host: " << std::endl;
        print_view(c_host);

        // then verify c as an output
        // TODO: this is going to fail because of a lack of parallel reductions!
        // this need to be reenabled once that has been solved!
        // for (size_t i = 0; i < N; i++)
        //     REQUIRE_THAT(c_host(i), Catch::Matchers::WithinRel(c_truth[i], 1.e-10));

        // done!
    }
    Kokkos::finalize();
}
#endif