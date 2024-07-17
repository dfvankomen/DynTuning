#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"
#include "data_transfers.hpp"
#include "kernel_matmatmult.hpp"
#include "kernel_matmul_eigenkokkos.hpp"
#include "kernel_matmul_kernels.hpp"
#include "kernel_matvecmult.hpp"
#include "kernel_testr0.hpp"
#include "kernel_vectordot.hpp"
#include "kernel_vectorouter.hpp"
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
        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k                      = KernelVectorDot<KernelHyperparameters>(options,
                                                        std::as_const(a_views),
                                                        std::as_const(b_views),
                                                        c_views);

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
TEST_CASE("Kernel: Verify MatVecMult Host and Device", "kernel")
{
    const size_t N = 10;

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

        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k                      = KernelMatVecMult<KernelHyperparameters>(options,
                                                         std::as_const(A_views),
                                                         std::as_const(b_views),
                                                         c_views);

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

        // then verify c as an output
        // TODO: this is going to fail because of a lack of parallel reductions!
        // this need to be reenabled once that has been solved!
        for (size_t i = 0; i < N; i++)
            REQUIRE_THAT(c_host(i), Catch::Matchers::WithinRel(c_truth[i], 1.e-10));

        // done!
    }
    Kokkos::finalize();
}

TEST_CASE("Kernel: Verify MatVecMult Non-Square Host and Device", "kernel")
{
    const size_t N = 10;
    const size_t M = 15;

    Kokkos::initialize();
    {
        DynMatrix2D A(N, M);
        std::vector<double> b(M);
        std::vector<double> c(M);
        std::vector<double> c_truth(M, 0.0);

        // initialize the A matrix
        int ij = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
                A(i, j) = static_cast<double>(ij++);

        // b can just be all 10's
        for (size_t i = 0; i < M; i++)
            b[i] = 10.0;

        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
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

        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k                      = KernelMatVecMult<KernelHyperparameters>(options,
                                                         std::as_const(A_views),
                                                         std::as_const(b_views),
                                                         c_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < M; i++)
            REQUIRE_THAT(c_host(i), Catch::Matchers::WithinRel(c_truth[i], 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < M; i++)
            c[i] = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < M; i++)
            REQUIRE_THAT(c_host(i), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);

        // then verify c as an output
        // TODO: this is going to fail because of a lack of parallel reductions!
        // this need to be reenabled once that has been solved!
        for (size_t i = 0; i < M; i++)
            REQUIRE_THAT(c_host(i), Catch::Matchers::WithinRel(c_truth[i], 1.e-10));

        // done!
    }
    Kokkos::finalize();
}


TEST_CASE("Kernel: Verify Square VectorOuter Host and Device", "kernel")
{
    const size_t N = 10;

    Kokkos::initialize();
    {
        std::vector<double> a(N);
        std::vector<double> b(N);
        DynMatrix2D C(N, N);
        DynMatrix2D C_truth(N, N);

        std::iota(a.begin(), a.end(), 1.0);
        std::iota(b.begin(), b.end(), -10.0);

        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
            {
                C_truth(i, j) = a[i] * b[j];
            }

        constexpr auto data_names = std::make_tuple(HashedName<hash("a")>(),
                                                    HashedName<hash("b")>(),
                                                    HashedName<hash("C")>());

        auto data_views = create_views(pack(a, b, C));

        auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);
        auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);
        auto& C_views = std::get<find<hash("C")>(data_names)>(data_views);

        // used for the checks after we finish
        auto& a_host = std::get<0>(a_views);
        auto& b_host = std::get<0>(b_views);
        auto& C_host = std::get<0>(C_views);

        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k                      = KernelVectorOuter<KernelHyperparameters>(options,
                                                          std::as_const(a_views),
                                                          std::as_const(b_views),
                                                          C_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                C(i, j) = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // done!
        // print_view(a_host);
        // print_view(b_host);
        // print_view(C_host);
    }
    Kokkos::finalize();
}


TEST_CASE("Kernel: Verify Non-Square VectorOuter Host and Device", "kernel")
{
    const size_t N = 10;
    const size_t M = 15;

    Kokkos::initialize();
    {
        std::vector<double> a(N);
        std::vector<double> b(M);
        DynMatrix2D C(N, M);
        DynMatrix2D C_truth(N, M);

        std::iota(a.begin(), a.end(), 1.0);
        std::iota(b.begin(), b.end(), -10.0);

        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
            {
                C_truth(i, j) = a[i] * b[j];
            }

        constexpr auto data_names = std::make_tuple(HashedName<hash("a")>(),
                                                    HashedName<hash("b")>(),
                                                    HashedName<hash("C")>());

        auto data_views = create_views(pack(a, b, C));

        auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);
        auto& b_views = std::get<find<hash("b")>(data_names)>(data_views);
        auto& C_views = std::get<find<hash("C")>(data_names)>(data_views);

        // used for the checks after we finish
        auto& a_host = std::get<0>(a_views);
        auto& b_host = std::get<0>(b_views);
        auto& C_host = std::get<0>(C_views);

        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k                      = KernelVectorOuter<KernelHyperparameters>(options,
                                                          std::as_const(a_views),
                                                          std::as_const(b_views),
                                                          C_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
                C(i, j) = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // done!
        // print_view(a_host);
        // print_view(b_host);
        // print_view(C_host);
    }
    Kokkos::finalize();
}

TEST_CASE("Kernel: Verify Square MatMatMult Host and Device", "kernel")
{
    const size_t N = 10;

    Kokkos::initialize();
    {
        DynMatrix2D A(N, N);
        DynMatrix2D B(N, N);
        DynMatrix2D C(N, N);
        DynMatrix2D C_truth(N, N);

        // initialize the A and B matrices!
        int ij = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
            {
                A(i, j) = static_cast<double>(ij++);
                B(i, j) = static_cast<double>(ij++);
            }


        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
            {
                double sum = 0.0;
                for (size_t k = 0; k < N; k++)
                    sum += A(i, k) * B(k, j);
                C_truth(i, j) = sum;
            }


        constexpr auto data_names = std::make_tuple(HashedName<hash("A")>(),
                                                    HashedName<hash("B")>(),
                                                    HashedName<hash("C")>());

        auto data_views = create_views(pack(A, B, C));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);
        auto& C_views = std::get<find<hash("C")>(data_names)>(data_views);

        // used for the checks after we finish
        auto& A_host = std::get<0>(A_views);
        auto& B_host = std::get<0>(B_views);
        auto& C_host = std::get<0>(C_views);


        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k                      = KernelMatMatMult<KernelHyperparameters>(options,
                                                         std::as_const(A_views),
                                                         std::as_const(B_views),
                                                         C_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                C(i, j) = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // done!
    }
    Kokkos::finalize();
}


TEST_CASE("Kernel: Verify Non-Square MatMatMult Host and Device", "kernel")
{
    const size_t N = 3;
    const size_t M = 4;
    const size_t O = 5;

    Kokkos::initialize();
    {
        DynMatrix2D A(N, M);
        DynMatrix2D B(M, O);
        DynMatrix2D C(N, O);
        DynMatrix2D C_truth(N, O);

        // initialize the A and B matrices!
        int ij = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
                A(i, j) = static_cast<double>(ij++);

        ij = -100;
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < O; j++)
                B(i, j) = static_cast<double>(ij++);

        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
            {
                double sum = 0.0;
                for (size_t k = 0; k < M; k++)
                    sum += A(i, k) * B(k, j);
                C_truth(i, j) = sum;
            }

        constexpr auto data_names = std::make_tuple(HashedName<hash("A")>(),
                                                    HashedName<hash("B")>(),
                                                    HashedName<hash("C")>());

        auto data_views = create_views(pack(A, B, C));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);
        auto& C_views = std::get<find<hash("C")>(data_names)>(data_views);

        // used for the checks after we finish
        auto& A_host = std::get<0>(A_views);
        auto& B_host = std::get<0>(B_views);
        auto& C_host = std::get<0>(C_views);


        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k                      = KernelMatMatMult<KernelHyperparameters>(options,
                                                         std::as_const(A_views),
                                                         std::as_const(B_views),
                                                         C_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                C(i, j) = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // done!
    }
    Kokkos::finalize();
}


TEST_CASE("Kernel: Verify Kernel w/ Rank 0", "kernel")
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
        auto k = KernelTestR0(options, std::as_const(a_views), std::as_const(b_views), c_views);

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


TEST_CASE("Kernel: Verify Square gemm Kokkos-Kernels", "kernel")
{
    const size_t N = 10;

    Kokkos::initialize();
    {
        DynMatrix2D A(N, N);
        DynMatrix2D B(N, N);
        DynMatrix2D C(N, N);
        DynMatrix2D C_truth(N, N);

        // initialize the A and B matrices!
        int ij = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
            {
                A(i, j) = static_cast<double>(ij++);
                B(i, j) = static_cast<double>(ij++);
            }

        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
            {
                double sum = 0.0;
                for (size_t k = 0; k < N; k++)
                    sum += A(i, k) * B(k, j);
                C_truth(i, j) = sum;
            }


        constexpr auto data_names = std::make_tuple(HashedName<hash("A")>(),
                                                    HashedName<hash("B")>(),
                                                    HashedName<hash("C")>());

        auto data_views = create_views(pack(A, B, C));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);
        auto& C_views = std::get<find<hash("C")>(data_names)>(data_views);

        // used for the checks after we finish
        auto& A_host = std::get<0>(A_views);
        auto& B_host = std::get<0>(B_views);
        auto& C_host = std::get<0>(C_views);


        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        auto k = KernelMatMulKokkosKernel(options,
                                          std::as_const(A_views),
                                          std::as_const(B_views),
                                          C_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                C(i, j) = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // done!
    }
    Kokkos::finalize();
}


TEST_CASE("Kernel: Verify Non-Square gemm Kokkos-Kernels", "kernel")
{
    const size_t N = 3;
    const size_t M = 4;
    const size_t O = 5;

    Kokkos::initialize();
    {
        DynMatrix2D A(N, M);
        DynMatrix2D B(M, O);
        DynMatrix2D C(N, O);
        DynMatrix2D C_truth(N, O);

        A.unaryExpr([&](double x) { return 0.0; });

        // initialize the A and B matrices!
        int ij = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
            {
                A(i, j) = static_cast<double>(ij++);
            }

        ij = -100;
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < O; j++)
                B(i, j) = static_cast<double>(ij++);

        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
            {
                double sum = 0.0;
                for (size_t k = 0; k < M; k++)
                    sum += A(i, k) * B(k, j);
                C_truth(i, j) = sum;
            }

        constexpr auto data_names = std::make_tuple(HashedName<hash("A")>(),
                                                    HashedName<hash("B")>(),
                                                    HashedName<hash("C")>());

        auto data_views = create_views(pack(A, B, C));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);
        auto& C_views = std::get<find<hash("C")>(data_names)>(data_views);

        // used for the checks after we finish
        auto& A_host = std::get<0>(A_views);
        auto& B_host = std::get<0>(B_views);
        auto& C_host = std::get<0>(C_views);


        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        auto k = KernelMatMulKokkosKernel(options,
                                          std::as_const(A_views),
                                          std::as_const(B_views),
                                          C_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                C(i, j) = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // done!
    }
    Kokkos::finalize();
}



TEST_CASE("Kernel: Verify Square gemm Eigen+KokkosKernel", "kernel")
{
    const size_t N = 10;

    Kokkos::initialize();
    {
        DynMatrix2D A(N, N);
        DynMatrix2D B(N, N);
        DynMatrix2D C(N, N);
        DynMatrix2D C_truth(N, N);

        // initialize the A and B matrices!
        int ij = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
            {
                A(i, j) = static_cast<double>(ij++);
                B(i, j) = static_cast<double>(ij++);
            }

        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
            {
                double sum = 0.0;
                for (size_t k = 0; k < N; k++)
                    sum += A(i, k) * B(k, j);
                C_truth(i, j) = sum;
            }


        constexpr auto data_names = std::make_tuple(HashedName<hash("A")>(),
                                                    HashedName<hash("B")>(),
                                                    HashedName<hash("C")>());

        auto data_views = create_views(pack(A, B, C));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);
        auto& C_views = std::get<find<hash("C")>(data_names)>(data_views);

        // used for the checks after we finish
        auto& A_host = std::get<0>(A_views);
        auto& B_host = std::get<0>(B_views);
        auto& C_host = std::get<0>(C_views);


        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        auto k = KernelMatMulEigenKokkosKernel(options,
                                               std::as_const(A_views),
                                               std::as_const(B_views),
                                               C_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                C(i, j) = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // done!
    }
    Kokkos::finalize();
}


TEST_CASE("Kernel: Verify Non-Square gemm Eigen+KokkosKernel", "kernel")
{
    const size_t N = 3;
    const size_t M = 4;
    const size_t O = 5;

    Kokkos::initialize();
    {
        DynMatrix2D A(N, M);
        DynMatrix2D B(M, O);
        DynMatrix2D C(N, O);
        DynMatrix2D C_truth(N, O);

        A.unaryExpr([&](double x) { return 0.0; });

        // initialize the A and B matrices!
        int ij = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < M; j++)
            {
                A(i, j) = static_cast<double>(ij++);
            }

        ij = -100;
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < O; j++)
                B(i, j) = static_cast<double>(ij++);

        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
            {
                double sum = 0.0;
                for (size_t k = 0; k < M; k++)
                    sum += A(i, k) * B(k, j);
                C_truth(i, j) = sum;
            }

        constexpr auto data_names = std::make_tuple(HashedName<hash("A")>(),
                                                    HashedName<hash("B")>(),
                                                    HashedName<hash("C")>());

        auto data_views = create_views(pack(A, B, C));

        auto& A_views = std::get<find<hash("A")>(data_names)>(data_views);
        auto& B_views = std::get<find<hash("B")>(data_names)>(data_views);
        auto& C_views = std::get<find<hash("C")>(data_names)>(data_views);

        // used for the checks after we finish
        auto& A_host = std::get<0>(A_views);
        auto& B_host = std::get<0>(B_views);
        auto& C_host = std::get<0>(C_views);


        KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

        // build the kernel
        auto k = KernelMatMulEigenKokkosKernel(options,
                                               std::as_const(A_views),
                                               std::as_const(B_views),
                                               C_views);

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // TODO: only call this if CUDA is enabled, probably

        // clear the c_host back to 0's
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                C(i, j) = 0.0;

        // verify that c_host is 0's (to be safe)
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinULP(0.0, 0));

        // then do the data transfer to device
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_host_to_device(i_view, k.data_views_);

        // then run it on device
        k(DeviceSelector::DEVICE);

        // then move it back to host for testing
        for (size_t i_view = 0; i_view < 3; i_view++)
            transfer_data_device_to_host(i_view, k.data_views_);

        // then verify c as an output
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < O; j++)
                REQUIRE_THAT(C_host(i, j), Catch::Matchers::WithinRel(C_truth(i, j), 1.e-10));

        // done!
    }
    Kokkos::finalize();
}