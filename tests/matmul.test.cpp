
#include "common.hpp"
#include "data.hpp"
#include "data_transfers.hpp"
#include "view.hpp"

// kernels
#include "kernel_matmul_3d.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_random.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <utility>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;
typedef Kokkos::MDRangePolicy<DeviceExecSpace, Kokkos::Rank<2>> device_rank2_range_policy;
typedef Kokkos::MDRangePolicy<DeviceExecSpace, Kokkos::Rank<3>> device_rank3_range_policy;


TEST_CASE("Kernel: Matmul3d", "[kernel][matmul]")
{
    const size_t N = 4;
    const size_t M = 5;
    const size_t K = 10;
    const size_t P = 3;

    // Eigen input matrices
    Eigen::MatrixXd A = Eigen::MatrixXd::Random(P, N);
    Eigen::Tensor<double, 3> X(N, M, K);
    X.setRandom();
    Eigen::Tensor<double, 3> X_out(P, M, K);
    Eigen::Tensor<double, 3> X_out_truth(P, M, K);

    Eigen::array<Eigen::IndexPair<int>, 1> contraction_dims = { Eigen::IndexPair<int>(1, 0) };

    // compute the truth now
    for (int k = 0; k < K; ++k)
    {
        Eigen::Tensor<double, 2> X_slice_2d = X.chip(k, 2);
        Eigen::TensorMap<Eigen::Tensor<const double, 2>> A_tensor(A.data(), P, N);
        X_out_truth.chip(k, 2) = A_tensor.contract(X_slice_2d, contraction_dims);

        for (int i = 0; i < P; ++i)
            for (int j = 0; j < M; ++j)
                X_out(i, j, k) = -999999999.9;
    }

    // create the views
    constexpr auto data_names = std::make_tuple(HashedName<hash("A")>(),
                                                HashedName<hash("X")>(),
                                                HashedName<hash("X_out")>());

    auto data_views = create_views(pack(A, X, X_out));

    auto& A_views     = std::get<find<hash("A")>(data_names)>(data_views);
    auto& X_views     = std::get<find<hash("X")>(data_names)>(data_views);
    auto& X_out_views = std::get<find<hash("X_out")>(data_names)>(data_views);

    auto& X_out_host = std::get<0>(X_out_views);


    // kernel options
    KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    using ChosenLinspace        = NoOptions;
    using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;

    // kernel
    auto k = KernelMatMul3DBatched<KernelHyperparameters>(options,
                                                          std::as_const(A_views),
                                                          std::as_const(X_views),
                                                          X_out_views);

    // run on host
    k(DeviceSelector::HOST);

    // verify the output
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < P; ++i)
            for (int j = 0; j < M; ++j)
                REQUIRE_THAT(X_out_host(i, j, k),
                             Catch::Matchers::WithinRel(X_out_truth(i, j, k), 1e-10));

    // then set back to bad values in x_out
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < P; ++i)
            for (int j = 0; j < M; ++j)
                X_out_host(i, j, k) = -999999999.9;

    // transfer data to device
    for (size_t i_view = 0; i_view < 3; i_view++)
        transfer_data_host_to_device(i_view, k.data_views_);

    // then run it
    k(DeviceSelector::DEVICE);

    // then move it back
    for (size_t i_view = 0; i_view < 3; i_view++)
        transfer_data_device_to_host(i_view, k.data_views_);

    // verify again
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < P; ++i)
            for (int j = 0; j < M; ++j)
                REQUIRE_THAT(X_out_host(i, j, k),
                             Catch::Matchers::WithinRel(X_out_truth(i, j, k), 1e-10));
}
