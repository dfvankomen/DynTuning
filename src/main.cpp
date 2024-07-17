// Assumptions:
// * Algorithms are cast as operating on one element at a time
// * Kernels are steps in the algorithm that typically involve a loop over the elements
// * Inputs and outputs are all ndarrays at the element level
//

#include "Kokkos_Core.hpp"
#include "algorithm.hpp"
#include "common.hpp"
#include "data.hpp"
#include "kernel_matvecmult.hpp"
#include "kernel_vectordot.hpp"

#include <fstream>
#include <istream>

/*
#define init_data_views(...) \
    auto data_views = create_views(pack(__VA_ARGS__));

#define get_data_view_macro(_1,_2,NAME,...) NAME
#define get_data_view_1(x) \
    std::get<find<hash(x)>(data_names)>(data_views)
#define get_data_view_2(x, device) \
    std::get<device>(std::get<find<hash(x)>(data_names)>(data_views))
#define get_data_view(...) get_data_view_macro(__VA_ARGS__, get_data_view_2,
get_data_view_1)(__VA_ARGS__) #define get_data_view_host(x) get_data_view(x, 0) #define
get_data_view_device(x) get_data_view(x, 1) #define get_data_view_scratch(x) get_data_view(x, 2)
*/

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[])
{

    // seed any randomization!
    std::srand(unsigned(std::time(0)));

    DeviceSelector device   = set_device(argc, argv);
    int N                   = set_N(argc, argv);
    bool reordering         = set_reordering(argc, argv);
    bool initialize         = set_initialize(argc, argv);
    int num_sims            = set_num_sims(argc, argv);
    int num_chain_runs      = set_num_chain_runs(argc, argv);
    int num_output_truncate = set_num_output_truncate(argc, argv);

#ifdef DYNTUNE_SINGLE_CHAIN_RUN
    unsigned int single_chain = set_single_chain_run(argc, argv);
#endif

    // execution options
    KernelOptions options;
    if (device != DeviceSelector::AUTO)
        options = { { device } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    // Initialize Kokkos
    printf("\nInitializing Kokkos...");
    Kokkos::initialize();
    printf("\nKokkos initialized!\n");
    { // start Kokkos scope

        // first define all data
        std::vector<double> q(N);
        std::vector<double> r(N);
        std::vector<double> s(N);
        DynMatrix2D t(N, N);
        std::vector<double> u(N);
        std::vector<double> v(N);
        std::vector<double> w(N);
        std::vector<double> x(N);
        std::vector<double> y(N);
        std::vector<double> z(N);

        std::vector<double> s_truth(N);
        std::vector<double> w_truth(N);
        std::vector<double> z_truth(N);

        // initialize data
        if (initialize)
        {
            printf("\ninitializing data\n");
            std::iota(q.begin(), q.end(), 1.0);
            std::iota(r.begin(), r.end(), 1.0);
            std::iota(s.begin(), s.end(), 1.0);
            {
                int ij = 0;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                        t(i, j) = static_cast<double>(ij++);
            }
            std::iota(v.begin(), v.end(), 1.0);
            std::iota(u.begin(),
                      u.end(),
                      1.0); // temporary: initialized so we can disable the 2D cases
            std::iota(x.begin(), x.end(), 1.0);
            std::iota(y.begin(), y.end(), 1.0);


            for (size_t i = 0; i < N; i++)
            {
                s_truth[i] = q[i] * r[i];
                w_truth[i] = v[i] * u[i];
                z_truth[i] = s_truth[i] * y[i];
            }
        }

        // register all data into a DataManager
        // Has to be constexpr to use later in a constexpr invocation of find()
        // I have not found a cleaner way to write this...
        // I feel like some kind of recursive function or macro combo should work...
        constexpr auto data_names = std::make_tuple(HashedName<hash("q")>(),
                                                    HashedName<hash("r")>(),
                                                    HashedName<hash("s")>(),
                                                    HashedName<hash("t")>(),
                                                    HashedName<hash("u")>(),
                                                    HashedName<hash("v")>(),
                                                    HashedName<hash("w")>(),
                                                    HashedName<hash("x")>(),
                                                    HashedName<hash("y")>(),
                                                    HashedName<hash("z")>());
        /*
        init_data_views(q, r, s, t, u, v, w, x, y, z);
        auto& q_views = get_data_view("q");
        auto& r_views = get_data_view("r");
        auto& s_views = get_data_view("s");
        auto& t_views = get_data_view("t");
        auto& u_views = get_data_view("u");
        auto& v_views = get_data_view("v");
        auto& w_views = get_data_view("w");
        auto& x_views = get_data_view("x");
        auto& y_views = get_data_view("y");
        auto& z_views = get_data_view("z");
        */
        printf("\nbuilding views\n");
        auto data_views = create_views(pack(q, r, s, t, u, v, w, x, y, z));

        auto& q_views = std::get<find<hash("q")>(data_names)>(data_views);
        auto& r_views = std::get<find<hash("r")>(data_names)>(data_views);
        auto& s_views = std::get<find<hash("s")>(data_names)>(data_views);
        auto& t_views = std::get<find<hash("t")>(data_names)>(data_views);
        auto& u_views = std::get<find<hash("u")>(data_names)>(data_views);
        auto& v_views = std::get<find<hash("v")>(data_names)>(data_views);
        auto& w_views = std::get<find<hash("w")>(data_names)>(data_views);
        auto& x_views = std::get<find<hash("x")>(data_names)>(data_views);
        auto& y_views = std::get<find<hash("y")>(data_names)>(data_views);
        auto& z_views = std::get<find<hash("z")>(data_names)>(data_views);


        // TEMP: this is a temporary helper to try and figure out what's going on here
        iter_tuple(data_views,
                   [&]<typename TempViewType>(size_t view_id, TempViewType& temp_view)
        {
            std::cout << "View ID: " << view_id << std::endl;
            iter_tuple(temp_view,
                       [&]<typename DeviceHostViewTypeThing>(size_t view_inner_id,
                                                             DeviceHostViewTypeThing& temp_inner) {
                std::cout << "    inner: " << view_inner_id << " rank: " << temp_inner.rank()
                          << std::endl;
            });
        });
        // throw std::exception();


        // define all kernels
        printf("\nbuilding kernels\n");
        auto k1 = KernelVectorDot(options,
                                  std::as_const(q_views),
                                  std::as_const(r_views),
                                  s_views); // VectorDot
        auto k2 = KernelMatVecMult(options,
                                   std::as_const(t_views),
                                   std::as_const(r_views),
                                   x_views); // matvecmult
        auto k3 = KernelVectorDot(options,
                                  std::as_const(v_views),
                                  std::as_const(u_views),
                                  w_views); // VectorDot
        // auto k5 = KernelVectorDot(options, std::as_const(v_views), std::as_const(u_views),
        // w_views); // VectorDot
        auto k4 = KernelVectorDot(options,
                                  std::as_const(s_views),
                                  std::as_const(y_views),
                                  z_views); // VectorDot

        // register all kernels info an Algorithm
        // auto kernels = pack(k1, k2, k3, k4);
        printf("\nbuilding algorithm\n");
        auto kernels = pack(k1, k3, k4);
        // auto kernels = pack(k2);
        Algorithm algo(kernels, data_views, reordering);
        algo.set_num_chain_runs(num_chain_runs);



#ifdef DYNTUNE_SINGLE_CHAIN_RUN
        algo.set_selected_chain(single_chain);
#endif

        algo.set_validation_function([&s, &w, &z, &s_truth, &w_truth, &z_truth, &N]()
        {
#ifdef DYNTUNE_DEBUG_ENABLED_TEST
            std::cout << "Inside validation function! " << std::endl;

            double abs_difference = 0.0;

            // NOTE: verification of S depends on the device transfer back over since it's a
            // "dependent"
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(s[i] - s_truth[i]);
                // std::cout << s[i] << "," << s_truth[i] << " ";
                s[i] = 0.0;
            }
            // std::cout << std::endl;
            std::cout << "abs difference for s (output kernel 0): " << abs_difference / N
                      << std::endl;


            abs_difference = 0.0;
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(w[i] - w_truth[i]);
                // std::cout << w[i] << "," << w_truth[i] << " ";
                w[i] = 0.0;
            }
            // std::cout << std::endl;
            std::cout << "abs difference for w (output kernel 1): " << abs_difference / N
                      << std::endl;

            abs_difference = 0.0;
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(z[i] - z_truth[i]);
                // std::cout << z[i] << "," << z_truth[i] << " ";
                z[i] = 0.0;
            }
            // std::cout << std::endl;
            std::cout << "abs difference for z (output kernel 2): " << abs_difference / N
                      << std::endl
                      << std::endl;
#endif
        });

        // run the algorithm;
        printf("\nrunning algorithm...\n");

        double progress = 0.0;

        printf("Running algorithm %d times...", num_sims);
        for (size_t ii = 0; ii < num_sims; ii++)
            algo();

        std::cout << std::endl;

        algo.print_results(true, false, num_output_truncate, std::cout);

        // open up a file
        std::ofstream fileStream;
        fileStream.open("sample.csv");
        algo.print_results(true, true, std::numeric_limits<std::size_t>::max(), fileStream);
        fileStream.close();

        // TESTS
        // TestVectorDot(k1, q, r, s);
        // TestMatVecMult(k2, t, s, u);
        // TestVectorDot(k3, v, u, w);
        // TestVectorDot(k4, x, y, z);

    } // end Kokkos scope
    Kokkos::finalize();

    printf("\n** GRACEFUL EXIT **\n");
}
