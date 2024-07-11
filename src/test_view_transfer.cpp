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
#include "data_transfers.hpp"
#include "kernel.hpp"
#include "kernels/kernel_matvecmult.hpp"

#include <cstddef>
#include <tuple>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;

#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))


// now to create a tuple that stores a bunch of "classes" or types of some kind
// template<typename T>

// this is how the linspace is computed and stored in kernels,
// this will need to be modified



int main(int argc, char* argv[])
{
    std::srand(unsigned(std::time(0)));

    DeviceSelector device = set_device(argc, argv);
    int N                 = set_N(argc, argv);


    // std::cout << "Hey" << std::endl;
    // iter_tuple(tup, [&]<typename TupleType>(size_t id, TupleType& tup_val) {
    //     std::cout << id << ": " << tup_val << std::endl;
    // });

    Kokkos::initialize();
    { // kokkos scope

        // Eigen::MatrixXd x(N, N);
        Eigen::MatrixXd y(N, N);
        Eigen::MatrixXd z(N, N);
        // std::vector<double> a(N);
        std::vector<double> x(N);
        // std::vector<double> y(N);

        // attempt to build a kernel

        DynMatrix2D A(N, N);
        std::vector<double> b(N);
        std::vector<double> c(N);
        std::vector<double> c_truth(N, 0.0);

        std::cout << "now initializing the data..." << std::endl;
        // initialize the A matrix
        int ij = 0;
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                A(i, j) = static_cast<double>(ij++);
        std::cout << "... A initialized!" << std::endl;

        // b can just be all 10's
        for (size_t i = 0; i < N; i++)
            b[i] = 10.0;
        std::cout << "... b initialized!" << std::endl;

        // calculate the truth
        for (size_t i = 0; i < N; i++)
            for (size_t j = 0; j < N; j++)
                c_truth[i] += A(i, j) * b[j];
        std::cout << "... c_truth initialized!" << std::endl;

        std::cout << "data initialized with values!" << std::endl;

        std::cout << "creating the data views.." << std::endl;
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
        auto k = KernelMatVecMult(options, std::as_const(A_views), std::as_const(b_views), c_views);

        // now we have a kernel, and it's the matvec one
        // try running the kernel?
        std::cout << "Kernel device execution policy size: " << k.n_device_execution_policies_
                  << std::endl;

        // then run the kernel, starting on the host
        k(DeviceSelector::HOST);

        // then verify c as an output
        double error = 0.0;
        for (size_t i = 0; i < N; i++)
            error += std::abs(c_host(i) - c_truth[i]);
        std::cout << "Error host: " << error << std::endl;

        for (std::size_t ii = 0; ii < k.n_device_execution_policies_; ii++)
        {
            std::cout << "II = " << ii << std::endl;
            // clear the c_host back to 0's
            for (size_t i = 0; i < N; i++)
                c[i] = 0.0;

            error = 0.0;
            for (size_t i = 0; i < N; i++)
                error += std::abs(c_host(i) - c_truth[i]);
            std::cout << "    Error we've zeroed: " << error << std::endl;


            // then do the data transfer to device
            for (size_t i_view = 0; i_view < 3; i_view++)
                transfer_data_host_to_device(i_view, k.data_views_);

            // then run it on device
            k(DeviceSelector::DEVICE, ii);

            // then move it back to host for testing
            for (size_t i_view = 0; i_view < 3; i_view++)
                transfer_data_device_to_host(i_view, k.data_views_);

            // then verify c as an output
            // TODO: this is going to fail because of a lack of parallel reductions!
            // this need to be reenabled once that has been solved!
            error = 0.0;
            for (size_t i = 0; i < N; i++)
                error += std::abs(c_host(i) - c_truth[i]);
            if (error > 0.01)
            {
                std::cout << "\e[0;31m" << "    ERROR IS NOT GREAT!" << "\e[0m" << std::endl;
            }
            std::cout << "    Error device: " << error << std::endl;
        }


#if 0
        {
            printf("\ninitializing data\n");
            int ij = 0;
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                {
                    // x(i, j) = static_cast<double>(ij++);
                    // y(i, j) = static_cast<double>(ij++);
                    // "truth" vector, lets us do some transforming between x and y to see if data
                    // transfer worked
                    // z(i, j) = x(i, j) + y(i, j);
                }

            std::iota(x.begin(), x.end(), 1.0);
            // std::iota(y.begin(), y.end(), 120.0);
        }

        constexpr auto data_names = std::make_tuple(HashedName<hash("x")>(), HashedName<hash("y")>()
                                                    // HashedName<hash("a")>()
        );

        printf("\nbuilding views\n");

        auto data_views = create_views(pack(x, y));


        auto& x_views = std::get<find<hash("x")>(data_names)>(data_views);
        auto& y_views = std::get<find<hash("y")>(data_names)>(data_views);
        // auto& a_views = std::get<find<hash("a")>(data_names)>(data_views);


        // auto views_dimension_flopped = std::make_tuple(std::make_tuple(get_v(0, 0, data_views),
        //                                                                get_v(1, 0, data_views),
        //                                                                get_v(2, 0, data_views)),
        //                                                std::make_tuple(get_v(0, 1, data_views),
        //                                                                get_v(1, 1, data_views),
        //                                                                get_v(2, 1, data_views)),
        //                                                std::make_tuple(get_v(0, 2, data_views),
        //                                                                get_v(1, 2, data_views),
        //                                                                get_v(2, 2, data_views)));


        // invert the tuple with templated functions
        auto views_dimension_flopped = repack_views(data_views);


        auto& x_host = std::get<0>(x_views);
        std::cout << "x's first value is before transfer: " << x_host(0) << std::endl;

        // call the transfer data host to device
        for (size_t j_it = 0; j_it < 2; j_it++)
        {
            transfer_data_host_to_device(j_it, views_dimension_flopped);
        }

        // do some modification to the arrays
        auto& x_device = std::get<1>(x_views);
        // x_device(0) = 1020339.0;
        Kokkos::parallel_for("Loop1", device_range_policy(0, N), KOKKOS_LAMBDA(const int i) {
            x_device(i) = 500 + x_device(i);
        });


        // transfer back to host
        for (size_t j_it = 0; j_it < 2; j_it++)
        {
            transfer_data_device_to_host(j_it, views_dimension_flopped);
        }


        std::cout << "x's first value is after transfer: " << x_host(0) << std::endl;

#endif

    } // end of scope for kokkos (to remind us)
    Kokkos::finalize();

    printf("\n\n====  Finished! ====\n");
}