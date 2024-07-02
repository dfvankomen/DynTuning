#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"
#include "data_transfers.hpp"

#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;

#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))


int main(int argc, char* argv[])
{
    std::srand(unsigned(std::time(0)));

    DeviceSelector device = set_device(argc, argv);
    int N                 = set_N(argc, argv);

    Kokkos::initialize();
    { // kokkos scope

        // Eigen::MatrixXd x(N, N);
        Eigen::MatrixXd y(N, N);
        Eigen::MatrixXd z(N, N);
        // std::vector<double> a(N);
        std::vector<double> x(N);
        // std::vector<double> y(N);

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

    } // end of scope for kokkos (to remind us)
    Kokkos::finalize();

    printf("\n\n====  Finished! ====\n");
}