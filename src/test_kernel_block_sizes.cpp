/**
 * @file THIS FILE HAS BASICALLY DEVOLVED INTO STAGING GROUNDS FOR TESTING TO AVOID OVERCOMPILING
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-07-11
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"
#include "kernel.hpp"
#include "kernel_matmatmult.hpp"

#include <cstddef>
#include <decl/Kokkos_Declare_OPENMP.hpp>
#include <tuple>
#include <utility>
#include <vector>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;


int main(int argc, char* argv[])
{
    std::srand(unsigned(std::time(0)));

    DeviceSelector device = set_device(argc, argv);
    int N                 = set_N(argc, argv);

    // execution options
    KernelOptions options;
    if (device != DeviceSelector::AUTO)
        options = { { device } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    // std::cout << "Hey" << std::endl;
    // iter_tuple(tup, [&]<typename TupleType>(size_t id, TupleType& tup_val) {
    //     std::cout << id << ": " << tup_val << std::endl;
    // });

    Kokkos::initialize();
    { // kokkos scope

        Kokkos::Timer timer;

        // Eigen::MatrixXd x(N, N);
        Eigen::MatrixXd x(N, N);
        Eigen::MatrixXd y(N, N);
        Eigen::MatrixXd z(N, N);
        std::vector<double> x_vec(N);

        constexpr auto data_names = std::make_tuple(HashedName<hash("x")>(),
                                                    HashedName<hash("y")>(),
                                                    HashedName<hash("z")>(),
                                                    HashedName<hash("x_vec")>());

        auto data_views = create_views(pack(x, y, z, x_vec));

        auto& x_views     = std::get<find<hash("x")>(data_names)>(data_views);
        auto& y_views     = std::get<find<hash("y")>(data_names)>(data_views);
        auto& z_views     = std::get<find<hash("z")>(data_names)>(data_views);
        auto& x_vec_views = std::get<find<hash("x_vec")>(data_names)>(data_views);

        // go through all of the options and see how things change
        // build up the kernel
        const unsigned int minblocks = 1;

        const unsigned int num_total_run = 11;

        std::vector<double> timings(num_total_run, 0);

        // default kernel, uses the normal 0,0 build
        auto k0 = KernelMatMatMult<HyperparameterOptions<NoOptions>>(options,
                                                                     std::as_const(x_views),
                                                                     std::as_const(y_views),
                                                                     z_views);

        // list all of the kernel options we want to run...
        // REMEMBER: you need at least 32 to start!
        // auto k1 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<16, minblocks>>>(
        //   options,
        //   std::as_const(x_views),
        //   std::as_const(y_views),
        //   z_views);
        auto k1 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<32, minblocks>>>(
          options,
          std::as_const(x_views),
          std::as_const(y_views),
          z_views);
        auto k2 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<64, minblocks>>>(
          options,
          std::as_const(x_views),
          std::as_const(y_views),
          z_views);
        auto k3 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<128, minblocks>>>(
          options,
          std::as_const(x_views),
          std::as_const(y_views),
          z_views);
        auto k4 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<256, minblocks>>>(
          options,
          std::as_const(x_views),
          std::as_const(y_views),
          z_views);
        auto k5 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<512, minblocks>>>(
          options,
          std::as_const(x_views),
          std::as_const(y_views),
          z_views);
        auto k6 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<1024, minblocks>>>(
          options,
          std::as_const(x_views),
          std::as_const(y_views),
          z_views);
        auto k7 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<2048, minblocks>>>(
          options,
          std::as_const(x_views),
          std::as_const(y_views),
          z_views);
        auto k8 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<4096, minblocks>>>(
          options,
          std::as_const(x_views),
          std::as_const(y_views),
          z_views);
        auto k9 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<8192, minblocks>>>(
          options,
          std::as_const(x_views),
          std::as_const(y_views),
          z_views);
        auto k10 = KernelMatMatMult<HyperparameterOptions<SingleLaunchBound<16384, minblocks>>>(
          options,
          std::as_const(x_views),
          std::as_const(y_views),
          z_views);

        // then execute the device version and time
        DeviceSelector ds = DeviceSelector::DEVICE;

        timer.reset();
        Kokkos::fence();

        // run through each and put down a stopper and time
        timer.reset();
        k0(ds, 0);
        Kokkos::fence();
        timings[0] = timer.seconds();

        timer.reset();
        k1(ds, 0);
        Kokkos::fence();
        timings[1] = timer.seconds();

        timer.reset();
        k2(ds, 0);
        Kokkos::fence();
        timings[2] = timer.seconds();

        timer.reset();
        k3(ds, 0);
        Kokkos::fence();
        timings[3] = timer.seconds();

        timer.reset();
        k4(ds, 0);
        Kokkos::fence();
        timings[4] = timer.seconds();

        timer.reset();
        k5(ds, 0);
        Kokkos::fence();
        timings[5] = timer.seconds();

        timer.reset();
        k6(ds, 0);
        Kokkos::fence();
        timings[6] = timer.seconds();

        timer.reset();
        k7(ds, 0);
        Kokkos::fence();
        timings[7] = timer.seconds();

        timer.reset();
        k8(ds, 0);
        Kokkos::fence();
        timings[8] = timer.seconds();

        timer.reset();
        k9(ds, 0);
        Kokkos::fence();
        timings[9] = timer.seconds();

        timer.reset();
        k10(ds, 0);
        Kokkos::fence();
        timings[10] = timer.seconds();

        timer.reset();

        for (std::size_t i = 0; i < timings.size(); ++i)
        {
            std::cout << i << "," << timings[i] << std::endl;
        }

    } // end of scope for kokkos (to remind us)
    Kokkos::finalize();


    printf("\n\n====  Finished! ====\n");
}