// Assumptions:
// * Optimizers are cast as operating on one element at a time
// * Kernels are steps in the optimizer that typically involve a loop over the elements
// * Inputs and outputs are all ndarrays at the element level
//

// TODO: this file should be cleaned up, there are lots of unnecessary objects created for data that
// aren't needed This became basically a large testing ground for the framework. It shouldn't be
// considered "part" of the framework


#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"
#include "kernel.hpp"
#include "kernel_branch_testkernel.hpp"
#include "kernel_branch_testkernel2d.hpp"
#include "kernel_matmatmult.hpp"
#include "kernel_matsum.hpp"
#include "kernel_matvecmult.hpp"
#include "kernel_vectordot.hpp"
#include "kernel_vectorouter.hpp"
#include "kernel_xxderiv_2d.hpp"
#include "kernel_yyderiv_2d.hpp"
#include "optimizer.hpp"

#include <fstream>
#include <istream>
#include <utility>

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
    std::string save_prefix = set_save_prefix(argc, argv);

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

        // matrix-matrix tests!
        DynMatrix2D aa(N, N); // input
        DynMatrix2D bb(N, N); // input
        DynMatrix2D cc(N, N); // output
        DynMatrix2D dd(N, N); // input
        DynMatrix2D ee(N, N); // input
        DynMatrix2D ff(N, N); // output
        DynMatrix2D gg(N, N); // input
        DynMatrix2D hh(N, N); // output

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


            {
                int ij = 0;
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++)
                    {
                        aa(i, j) = static_cast<double>(ij++);
                        bb(i, j) = static_cast<double>(ij++);
                        dd(i, j) = static_cast<double>(ij++);
                        ee(i, j) = static_cast<double>(ij++);
                        gg(i, j) = static_cast<double>(ij++);
                    }
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
                                                    HashedName<hash("z")>(),
                                                    HashedName<hash("aa")>(),
                                                    HashedName<hash("bb")>(),
                                                    HashedName<hash("cc")>(),
                                                    HashedName<hash("dd")>(),
                                                    HashedName<hash("ee")>(),
                                                    HashedName<hash("ff")>(),
                                                    HashedName<hash("gg")>(),
                                                    HashedName<hash("hh")>());
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
        auto data_views =
          create_views(pack(q, r, s, t, u, v, w, x, y, z, aa, bb, cc, dd, ee, ff, gg, hh));

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


        auto& aa_views = std::get<find<hash("aa")>(data_names)>(data_views);
        auto& bb_views = std::get<find<hash("bb")>(data_names)>(data_views);
        auto& cc_views = std::get<find<hash("cc")>(data_names)>(data_views);
        auto& dd_views = std::get<find<hash("dd")>(data_names)>(data_views);
        auto& ee_views = std::get<find<hash("ee")>(data_names)>(data_views);
        auto& ff_views = std::get<find<hash("ff")>(data_names)>(data_views);
        auto& gg_views = std::get<find<hash("gg")>(data_names)>(data_views);
        auto& hh_views = std::get<find<hash("hh")>(data_names)>(data_views);


        // define all kernels
        printf("\nbuilding kernels\n");

        // using ChosenLinspace = LinspaceOptions<32, 512, 5, 1, 10, 3>;
        using ChosenLinspace = LinspaceOptions<32, 512, 16, 1, 100, 5>;

        using KernelHyperparameters  = HyperparameterOptions<NoOptions>;
        using KernelHyperparamters1D = HyperparameterOptions<NoOptions>;

        auto k1 = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(q_views),
                                                         std::as_const(r_views),
                                                         s_views); // VectorDot
        auto k2 = KernelMatVecMult<KernelHyperparameters>(options,
                                                          std::as_const(t_views),
                                                          std::as_const(r_views),
                                                          x_views); // matvecmult
        auto k3 = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(v_views),
                                                         std::as_const(u_views),
                                                         w_views); // VectorDot
        auto k5 = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(v_views),
                                                         std::as_const(u_views),
                                                         w_views); // VectorDot
        auto k4 = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(s_views),
                                                         std::as_const(y_views),
                                                         z_views); // VectorDot

        auto k_new1 = KernelMatMatMult<KernelHyperparameters>(options,
                                                              std::as_const(aa_views),
                                                              std::as_const(bb_views),
                                                              cc_views);
        auto k_new2 = KernelMatMatMult<KernelHyperparameters>(options,
                                                              std::as_const(dd_views),
                                                              std::as_const(ee_views),
                                                              ff_views);
        auto k_new3 = KernelMatMatMult<KernelHyperparameters>(options,
                                                              std::as_const(cc_views),
                                                              std::as_const(gg_views),
                                                              hh_views);

        auto k_gradx =
          KernelXXDeriv2D<KernelHyperparameters>(options, std::as_const(aa_views), dd_views);
        auto k_grady =
          KernelYYDeriv2D<KernelHyperparameters>(options, std::as_const(bb_views), ee_views);
        auto k_gradout = KernelMatSum<KernelHyperparameters>(options,
                                                             std::as_const(dd_views),
                                                             std::as_const(ee_views),
                                                             cc_views);


        auto k_conditional =
          KernelBranchTest<KernelHyperparamters1D>(options, std::as_const(q_views), r_views);

        auto k_conditional2d =
          KernelBranchTest2D<KernelHyperparameters>(options, std::as_const(aa_views), bb_views);


        // two test kernel chains
        auto k_first  = KernelVectorOuter<KernelHyperparameters>(options,
                                                                std::as_const(q_views),
                                                                std::as_const(r_views),
                                                                aa_views);
        auto k_second = KernelMatVecMult<KernelHyperparameters>(options,
                                                                std::as_const(aa_views),
                                                                std::as_const(s_views),
                                                                u_views);
        auto k_third =
          KernelBranchTest2D<KernelHyperparameters>(options, std::as_const(bb_views), cc_views);
        auto k_fourth = KernelVectorDot<KernelHyperparameters>(options,
                                                               std::as_const(v_views),
                                                               std::as_const(w_views),
                                                               x_views);

        // register all kernels info an Optimizer
        // auto kernels = pack(k1, k2, k3, k4);
        printf("\nbuilding optimizer\n");
        // auto kernels = pack(k1, k3, k4);
        // auto kernels = pack(k2);
        // auto kernels = pack(k_new1, k_new2, k_new3);
        // auto kernels = pack(k_grady);
        // auto kernels = pack(k_gradx);
        // auto kernels = pack(k_conditional);
        // auto kernels = pack(k1, k3);
        // auto kernels = pack(k_new1);
        // auto kernels = pack(k_conditional2d);
        // auto kernels = pack(k2);
        auto kernels = pack(k_first, k_second, k_third, k_fourth);

        Optimizer algo(kernels, data_views, reordering);
        algo.set_num_chain_runs(num_chain_runs);



#ifdef DYNTUNE_SINGLE_CHAIN_RUN
        algo.set_selected_chain(single_chain);
#endif

        // example validation function that can be set and run
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

        // run the optimizer;
        printf("\nrunning optimizer...\n");

        printf("Running optimizer %d times...", num_sims);
        for (size_t ii = 0; ii < num_sims; ii++)
        {
            printf("Now starting run %d\n", ii);
            algo();
            printf("  ...finished run %d\n", ii);
        }
        std::cout << std::endl;

        algo.print_results(true, false, num_output_truncate, std::cout);

        std::size_t full_output = std::numeric_limits<std::size_t>::max();

        // open up a file
        std::ofstream fileStream;
        fileStream.open(save_prefix + "results.csv");
        algo.print_results(true, true, full_output, fileStream);
        fileStream.close();

        // then we also need to dump the information about the chains so we can investigate it
        fileStream.open(save_prefix + "chains.txt");
        algo.dump_kernel_chains(fileStream, full_output);
        fileStream.close();

    } // end Kokkos scope
    Kokkos::finalize();

    printf("\n** GRACEFUL EXIT **\n");
}
