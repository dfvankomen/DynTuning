#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"
#include "data_transfers.hpp"
#include "kernel.hpp"

#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <utility>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;

#define get_v(i, j, tuple) std::get<j>(std::get<i>(tuple))

static inline std::size_t calc_linspace_inline(const std::size_t start,
                                               const std::size_t end,
                                               const std::size_t steps,
                                               const std::size_t i);

// linspace inline calculation
static inline std::size_t calc_linspace_inline(const std::size_t start,
                                               const std::size_t end,
                                               const std::size_t steps,
                                               const std::size_t i)
{
    // return (T)((double)start + (((double)end - (double)start) / (double)(steps - 1)) *
    // (double)i);
    return static_cast<std::size_t>(
      static_cast<double>(start) +
      ((static_cast<double>(end) - static_cast<double>(start)) / static_cast<double>(steps - 1)) *
        static_cast<double>(i));
}

template<std::size_t ThreadsStart,
         std::size_t ThreadsEnd,
         std::size_t ThreadsSteps,
         std::size_t... I>
inline static auto create_execution_spaces_first_step(std::integer_sequence<std::size_t, I...>)
{
    return std::make_tuple(calc_linspace_inline(ThreadsStart, ThreadsEnd, ThreadsSteps, I)...);
}

template<std::size_t ThreadsStart, std::size_t ThreadsEnd, std::size_t ThreadsSteps>
inline static auto create_execution_spaces_device()
{
    return create_execution_spaces_first_step<ThreadsStart, ThreadsEnd, ThreadsSteps>(
      std::make_index_sequence<ThreadsSteps> {});
}


// this is how the linspace is computed and stored in kernels,
// this will need to be modified
constexpr auto most_internal_linspace(double start, double end, std::size_t nsteps, std::size_t i)
{
    return start + i * ((end - start) / (nsteps - 1));
}

template<std::size_t... I>
constexpr auto linspace_calc(double start,
                             double end,
                             std::size_t nsteps,
                             std::index_sequence<I...>)
{
    return std::make_tuple(
      static_cast<std::size_t>(most_internal_linspace(start, end, nsteps, I))...);
}

template<std::size_t N>
constexpr auto linspace(double start, double end)
{
    static_assert(N > 1,
                  "Number of steps has to be greater than 1 for linspace calculations to work!");
    return linspace_calc(start, end, N, std::make_index_sequence<N> {});
}



// now to create a tuple that stores a bunch of "classes" or types of some kind
// template<typename T>

// this is how the linspace is computed and stored in kernels,
// this will need to be modified



int main(int argc, char* argv[])
{
    std::srand(unsigned(std::time(0)));

    DeviceSelector device = set_device(argc, argv);
    int N                 = set_N(argc, argv);

    // simple linspace tuple
    std::cout << std::endl << "1D Attempt, pure numbers:" << std::endl;
    auto mytuple    = linspace<10>(1, 52);
    auto printTuple = [](auto&&... args)
    {
        ((std::cout << args << " "), ...);
    };
    std::apply(printTuple, mytuple);
    std::cout << std::endl;

    // 1D with the kernel's max threads
    std::cout << std::endl << "1D Attempt:" << std::endl;
    auto mytuple_kernel   = linspace_kernel<10>(1, 20);
    auto printTupleKernel = [](auto&&... args)
    {
        ((std::cout << args.maxthreads << " "), ...);
    };
    std::apply(printTupleKernel, mytuple_kernel);
    std::cout << std::endl;


    std::cout << std::endl << "2D Attempt (tuple of tuples):" << std::endl;
    auto tuple_large = linspace_start<4, 5>(1, 10, 11, 20);
    iter_tuple(tuple_large,
               [&]<typename TupleType>(size_t i, TupleType& tup_val)
    {
        iter_tuple(tup_val, [&]<typename TupleTypeInner>(size_t j, TupleTypeInner& tupp) {
            std::cout << "(" << tupp.maxthreads << ", " << tupp.minblocks << ") ";
        });
    });
    std::cout << std::endl;

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