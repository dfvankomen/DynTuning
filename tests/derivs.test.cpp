#include "common.hpp"
#include "data.hpp"
#include "data_transfers.hpp"
#include "view.hpp"

// kernels
#include "kernel_xxderiv_3d.hpp"
#include "kernel_yyderiv_3d.hpp"
#include "kernel_zzderiv_3d.hpp"

#include <Kokkos_Core.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <utility>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;
typedef Kokkos::MDRangePolicy<DeviceExecSpace, Kokkos::Rank<2>> device_rank2_range_policy;
typedef Kokkos::MDRangePolicy<DeviceExecSpace, Kokkos::Rank<3>> device_rank3_range_policy;


// test function struct, this handles everything directly
struct TestFunctionIn3D
{
    double wx, wy, wz;

    TestFunctionIn3D(double wx_, double wy_, double wz_)
      : wx(wx_)
      , wy(wy_)
      , wz(wz_)
    {
    }

    double val(double x, double y, double z) const
    {
        return std::sin(wx * x) * std::sin(wy * y) * std::sin(wz * z);
    }

    double d2dx2(double x, double y, double z) const
    {
        return -wx * wx * val(x, y, z);
    }

    double d2dy2(double x, double y, double z) const
    {
        return -wy * wy * val(x, y, z);
    }

    double d2dz2(double x, double y, double z) const
    {
        return -wz * wz * val(x, y, z);
    }
};



TEST_CASE("Kernel: Verify XXDeriv3D Host and Device", "[kernel][deriv3d]")
{
    const double boundary_error_bound = 0.1;
    const double interior_error_bound = 1e-3;

    const size_t N = 64;
    const size_t M = 64;
    const size_t K = 64;

    const double Lx = 1.0;
    const double Ly = 0.9;
    const double Lz = 0.8;

    const double dx = (Lx - 0.0) / (N - 1.0);
    const double dy = (Ly - 0.0) / (M - 1.0);
    const double dz = (Lz - 0.0) / (K - 1.0);

    DynMatrix3D U(N, M, K);
    DynMatrix3D dUdxx_out(N, M, K);

    std::vector<double> spacing = { dx, dy, dz };

    TestFunctionIn3D f(2.0 * M_PI, 1.0 * M_PI, 0.5 * M_PI);

    // fill with the known function, which is sin(x)*cos(y)*sin(z) and it's analytic derivative
    for (size_t i = 0; i < N; ++i)
    {
        double x = std::lerp(0.0, Lx, i / (N - 1.0));
        for (size_t j = 0; j < M; ++j)
        {
            double y = std::lerp(0.0, Ly, j / (M - 1.0));
            for (size_t k = 0; k < K; ++k)
            {
                double z   = std::lerp(0.0, Lz, k / (K - 1.0));
                U(i, j, k) = f.val(x, y, z);
            }
        }
    }

    // create the views
    constexpr auto data_names = std::make_tuple(HashedName<hash("U")>(),
                                                HashedName<hash("spacing")>(),
                                                HashedName<hash("dUdxx")>());

    auto data_views = create_views(pack(U, spacing, dUdxx_out));

    auto& U_views       = std::get<find<hash("U")>(data_names)>(data_views);
    auto& dUdxx_views   = std::get<find<hash("dUdxx")>(data_names)>(data_views);
    auto& spacing_views = std::get<find<hash("spacing")>(data_names)>(data_views);

    auto& dUdxx_host = std::get<0>(dUdxx_views);


    KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    using ChosenLinspace        = NoOptions;
    using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;

    auto k = KernelXXDeriv3D<KernelHyperparameters>(options,
                                                    std::as_const(U_views),
                                                    std::as_const(spacing_views),
                                                    dUdxx_views);

    // run on host
    k(DeviceSelector::HOST);

    // verify the output
    for (size_t i = 0; i < N; ++i)
    {
        double x = std::lerp(0.0, Lx, i / (N - 1.0));
        for (size_t j = 0; j < M; ++j)
        {
            double y = std::lerp(0.0, Ly, j / (M - 1.0));
            for (size_t k = 0; k < K; ++k)
            {
                double z        = std::lerp(0.0, Lz, k / (K - 1.0));
                double expected = f.d2dx2(x, y, z);
                double actual   = dUdxx_host(i, j, k);

                double error = std::abs((actual - expected));

                if (i >= 2 && i < N - 2)
                {
                    REQUIRE(error < interior_error_bound);
                }
                else
                {
                    REQUIRE(error < boundary_error_bound);
                }
            }
        }
    }

    // then we can do the device, but first set all data on dUdzz_host to a bad value
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < M; ++j)
            for (size_t k = 0; k < K; ++k)
                dUdxx_host(i, j, k) = -999999999.9;

    for (size_t i_view = 0; i_view < 3; i_view++)
        transfer_data_host_to_device(i_view, k.data_views_);

    // then run it
    k(DeviceSelector::DEVICE);

    // then move it back
    for (size_t i_view = 0; i_view < 3; i_view++)
        transfer_data_device_to_host(i_view, k.data_views_);

    // verify the output
    for (size_t i = 0; i < N; ++i)
    {
        double x = std::lerp(0.0, Lx, i / (N - 1.0));
        for (size_t j = 0; j < M; ++j)
        {
            double y = std::lerp(0.0, Ly, j / (M - 1.0));
            for (size_t k = 0; k < K; ++k)
            {
                double z        = std::lerp(0.0, Lz, k / (K - 1.0));
                double expected = f.d2dx2(x, y, z);
                double actual   = dUdxx_host(i, j, k);

                double error = std::abs((actual - expected));

                if (i >= 2 && i < N - 2)
                {
                    REQUIRE(error < interior_error_bound);
                }
                else
                {
                    REQUIRE(error < boundary_error_bound);
                }
            }
        }
    }
}


TEST_CASE("Kernel: Verify YYDeriv3D Host and Device", "[kernel][deriv3d]")
{
    const double boundary_error_bound = 0.1;
    const double interior_error_bound = 1e-3;

    const size_t N = 64;
    const size_t M = 64;
    const size_t K = 64;

    const double Lx = 1.0;
    const double Ly = 0.9;
    const double Lz = 0.8;

    const double dx = (Lx - 0.0) / (N - 1.0);
    const double dy = (Ly - 0.0) / (M - 1.0);
    const double dz = (Lz - 0.0) / (K - 1.0);

    DynMatrix3D U(N, M, K);
    DynMatrix3D dUdyy_out(N, M, K);

    std::vector<double> spacing = { dx, dy, dz };

    TestFunctionIn3D f(2.0 * M_PI, 1.0 * M_PI, 0.5 * M_PI);

    // fill with the known function, which is sin(x)*cos(y)*sin(z) and it's analytic derivative
    for (size_t i = 0; i < N; ++i)
    {
        double x = std::lerp(0.0, Lx, i / (N - 1.0));
        for (size_t j = 0; j < M; ++j)
        {
            double y = std::lerp(0.0, Ly, j / (M - 1.0));
            for (size_t k = 0; k < K; ++k)
            {
                double z   = std::lerp(0.0, Lz, k / (K - 1.0));
                U(i, j, k) = f.val(x, y, z);
            }
        }
    }

    // create the views
    constexpr auto data_names = std::make_tuple(HashedName<hash("U")>(),
                                                HashedName<hash("spacing")>(),
                                                HashedName<hash("dUdyy")>());

    auto data_views = create_views(pack(U, spacing, dUdyy_out));

    auto& U_views       = std::get<find<hash("U")>(data_names)>(data_views);
    auto& dUdyy_views   = std::get<find<hash("dUdyy")>(data_names)>(data_views);
    auto& spacing_views = std::get<find<hash("spacing")>(data_names)>(data_views);

    auto& dUdyy_host = std::get<0>(dUdyy_views);


    KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    using ChosenLinspace        = NoOptions;
    using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;

    auto k = KernelYYDeriv3D<KernelHyperparameters>(options,
                                                    std::as_const(U_views),
                                                    std::as_const(spacing_views),
                                                    dUdyy_views);

    // run on host
    k(DeviceSelector::HOST);

    // verify the data
    for (size_t i = 0; i < N; ++i)
    {
        double x = std::lerp(0.0, Lx, i / (N - 1.0));
        for (size_t j = 0; j < M; ++j)
        {
            double y = std::lerp(0.0, Ly, j / (M - 1.0));
            for (size_t k = 0; k < K; ++k)
            {
                double z        = std::lerp(0.0, Lz, k / (K - 1.0));
                double expected = f.d2dy2(x, y, z);
                double actual   = dUdyy_host(i, j, k);

                double error = std::abs((actual - expected));

                if (j > 1 && j < M - 2)
                {
                    REQUIRE(error < interior_error_bound);
                }
                else
                {
                    REQUIRE(error < boundary_error_bound);
                }
            }
        }
    }


    // then we can do the device, but first set all data on dUdzz_host to a bad value
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < M; ++j)
            for (size_t k = 0; k < K; ++k)
                dUdyy_host(i, j, k) = -999999999.9;

    for (size_t i_view = 0; i_view < 3; i_view++)
        transfer_data_host_to_device(i_view, k.data_views_);

    // then run it
    k(DeviceSelector::DEVICE);

    // then move it back
    for (size_t i_view = 0; i_view < 3; i_view++)
        transfer_data_device_to_host(i_view, k.data_views_);

    // verify the data
    for (size_t i = 0; i < N; ++i)
    {
        double x = std::lerp(0.0, Lx, i / (N - 1.0));
        for (size_t j = 0; j < M; ++j)
        {
            double y = std::lerp(0.0, Ly, j / (M - 1.0));
            for (size_t k = 0; k < K; ++k)
            {
                double z        = std::lerp(0.0, Lz, k / (K - 1.0));
                double expected = f.d2dy2(x, y, z);
                double actual   = dUdyy_host(i, j, k);

                double error = std::abs((actual - expected));

                if (j > 1 && j < M - 2)
                {
                    REQUIRE(error < interior_error_bound);
                }
                else
                {
                    REQUIRE(error < boundary_error_bound);
                }
            }
        }
    }
}


TEST_CASE("Kernel: Verify ZZDeriv3D Host and Device", "[kernel][deriv3d]")
{
    const double boundary_error_bound = 0.1;
    const double interior_error_bound = 1e-3;

    const size_t N = 64;
    const size_t M = 64;
    const size_t K = 64;

    const double Lx = 1.0;
    const double Ly = 0.9;
    const double Lz = 0.8;

    const double dx = (Lx - 0.0) / (N - 1.0);
    const double dy = (Ly - 0.0) / (M - 1.0);
    const double dz = (Lz - 0.0) / (K - 1.0);

    DynMatrix3D U(N, M, K);
    DynMatrix3D dUdzz_out(N, M, K);

    std::vector<double> spacing = { dx, dy, dz };

    TestFunctionIn3D f(2.0 * M_PI, 1.0 * M_PI, 0.5 * M_PI);

    // fill with the known function, which is sin(x)*cos(y)*sin(z) and it's analytic derivative
    for (size_t i = 0; i < N; ++i)
    {
        double x = std::lerp(0.0, Lx, i / (N - 1.0));
        for (size_t j = 0; j < M; ++j)
        {
            double y = std::lerp(0.0, Ly, j / (M - 1.0));
            for (size_t k = 0; k < K; ++k)
            {
                double z   = std::lerp(0.0, Lz, k / (K - 1.0));
                U(i, j, k) = f.val(x, y, z);
            }
        }
    }

    // create the views
    constexpr auto data_names = std::make_tuple(HashedName<hash("U")>(),
                                                HashedName<hash("spacing")>(),
                                                HashedName<hash("dUdzz")>());

    auto data_views = create_views(pack(U, spacing, dUdzz_out));

    auto& U_views       = std::get<find<hash("U")>(data_names)>(data_views);
    auto& dUdzz_views   = std::get<find<hash("dUdzz")>(data_names)>(data_views);
    auto& spacing_views = std::get<find<hash("spacing")>(data_names)>(data_views);

    auto& dUdzz_host = std::get<0>(dUdzz_views);


    KernelOptions options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    using ChosenLinspace        = NoOptions;
    using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;

    auto k = KernelZZDeriv3D<KernelHyperparameters>(options,
                                                    std::as_const(U_views),
                                                    std::as_const(spacing_views),
                                                    dUdzz_views);

    // run on host
    k(DeviceSelector::HOST);

    // verify the data
    for (size_t i = 0; i < N; ++i)
    {
        double x = std::lerp(0.0, Lx, i / (N - 1.0));
        for (size_t j = 0; j < M; ++j)
        {
            double y = std::lerp(0.0, Ly, j / (M - 1.0));
            for (size_t k = 0; k < K; ++k)
            {
                double z        = std::lerp(0.0, Lz, k / (K - 1.0));
                double expected = f.d2dz2(x, y, z);
                double actual   = dUdzz_host(i, j, k);

                double error = std::abs((actual - expected));

                if (k > 1 && k < K - 2)
                {
                    REQUIRE(error < interior_error_bound);
                }
                else
                {
                    REQUIRE(error < boundary_error_bound);
                }
            }
        }
    }


    // then we can do the device, but first set all data on dUdzz_host to a bad value
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < M; ++j)
            for (size_t k = 0; k < K; ++k)
                dUdzz_host(i, j, k) = -999999999.9;

    for (size_t i_view = 0; i_view < 3; i_view++)
        transfer_data_host_to_device(i_view, k.data_views_);

    // then run it
    k(DeviceSelector::DEVICE);

    // then move it back
    for (size_t i_view = 0; i_view < 3; i_view++)
        transfer_data_device_to_host(i_view, k.data_views_);

    // verify the data
    for (size_t i = 0; i < N; ++i)
    {
        double x = std::lerp(0.0, Lx, i / (N - 1.0));
        for (size_t j = 0; j < M; ++j)
        {
            double y = std::lerp(0.0, Ly, j / (M - 1.0));
            for (size_t k = 0; k < K; ++k)
            {
                double z        = std::lerp(0.0, Lz, k / (K - 1.0));
                double expected = f.d2dz2(x, y, z);
                double actual   = dUdzz_host(i, j, k);

                double error = std::abs((actual - expected));

                if (k > 1 && k < K - 2)
                {
                    REQUIRE(error < interior_error_bound);
                }
                else
                {
                    REQUIRE(error < boundary_error_bound);
                }
            }
        }
    }
}
