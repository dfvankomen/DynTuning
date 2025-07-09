#include "common.hpp"
#include "data.hpp"
#include "kernel.hpp"
#include "kernel_nlsm.hpp"
#include "kernel_xxderiv_3d.hpp"
#include "kernel_yyderiv_3d.hpp"
#include "kernel_zzderiv_3d.hpp"
#include "optimizer.hpp"

#include <Kokkos_Core.hpp>

// include various kernels

#include <fstream>
#include <istream>
#include <utility>

int main(int argc, char* argv[])
{
    printf("Starting a NLSM run!\n");
    // seed any randomization!
    std::srand(unsigned(std::time(0)));

    // parameters
    DeviceSelector device   = set_device(argc, argv);
    int N                   = set_N(argc, argv);
    bool reordering         = set_reordering(argc, argv);
    bool initialize         = set_initialize(argc, argv);
    int num_sims            = set_num_sims(argc, argv);
    int num_chain_runs      = set_num_chain_runs(argc, argv);
    int num_output_truncate = set_num_output_truncate(argc, argv);
    std::string save_prefix = set_save_prefix(argc, argv);

    printf("\nParameters read, now initializing...\n");


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


#if 1
    { // start Kokkos scope
        const double dx = 0.01;

        // define the data that we'll be using
        DynMatrix3D phi(N, N, N);
        DynMatrix3D chi(N, N, N);
        DynMatrix3D grad2_0_0_chi(N, N, N);
        DynMatrix3D grad2_1_1_chi(N, N, N);
        DynMatrix3D grad2_2_2_chi(N, N, N);

        // outputs
        DynMatrix3D phi_rhs(N, N, N);
        DynMatrix3D chi_rhs(N, N, N);


        std::vector<double> spacing = { dx, dx, dx };

        // initialize the data so it's more interesting
        if (!initialize)
        {
            printf("\nInitializing data\n");

            const double amp   = 1.3;
            const double delta = 0.5;
            const double xc    = 0.0;
            const double yc    = 0.0;
            const double zc    = 0.0;
            const double epsx  = 1.0;
            const double epsy  = 1.0;
            const double epsz  = 1.0;
            const double R     = 1.0;

            for (size_t k = 0; k < N; k++)
            {
                for (size_t j = 0; j < N; j++)
                {
                    for (size_t i = 0; i < N; i++)
                    {
                        // calculate x y and z
                        double x = i * dx;
                        double y = j * dx;
                        double z = k * dx;

                        double rt = sqrt(epsx * (x - xc) * (x - xc) + epsy * (y - yc) * (y - yc) +
                                         epsz * (z - zc) * (z - zc));
                        double chi_temp = amp * exp(-(rt - R) * (rt - R) / (delta * delta));
                        double phi_temp = 0.0;

                        phi(k, j, i) = phi_temp;
                        chi(k, j, i) = chi_temp;
                    }
                }
            }
        }

        // then we register the data into the DataManager and all of that
        constexpr auto data_names = std::make_tuple(HashedName<hash("phi")>(),
                                                    HashedName<hash("chi")>(),
                                                    HashedName<hash("phi_rhs")>(),
                                                    HashedName<hash("chi_rhs")>(),
                                                    HashedName<hash("grad2_0_0_chi")>(),
                                                    HashedName<hash("grad2_1_1_chi")>(),
                                                    HashedName<hash("grad2_2_2_chi")>(),
                                                    HashedName<hash("spacing")>());

        // then build up views
        printf("\nbuilding views\n");
        auto data_views = create_views(
          pack(phi, chi, chi_rhs, chi_rhs, grad2_0_0_chi, grad2_1_1_chi, grad2_2_2_chi, spacing));

        auto& phi_views           = std::get<find<hash("phi")>(data_names)>(data_views);
        auto& chi_views           = std::get<find<hash("chi")>(data_names)>(data_views);
        auto& phi_rhs_views       = std::get<find<hash("phi_rhs")>(data_names)>(data_views);
        auto& chi_rhs_views       = std::get<find<hash("chi_rhs")>(data_names)>(data_views);
        auto& grad2_0_0_chi_views = std::get<find<hash("grad2_0_0_chi")>(data_names)>(data_views);
        auto& grad2_1_1_chi_views = std::get<find<hash("grad2_1_1_chi")>(data_names)>(data_views);
        auto& grad2_2_2_chi_views = std::get<find<hash("grad2_2_2_chi")>(data_names)>(data_views);
        auto& spacing_views       = std::get<find<hash("spacing")>(data_names)>(data_views);

        // now we define the kernels
        using KernelHyperparameters = HyperparameterOptions<NoOptions>;

        auto k_grad2_0_0_chi = KernelXXDeriv3D<KernelHyperparameters>(options,
                                                                      std::as_const(chi_views),
                                                                      std::as_const(spacing_views),
                                                                      grad2_0_0_chi_views);
        auto k_grad2_1_1_chi = KernelYYDeriv3D<KernelHyperparameters>(options,
                                                                      std::as_const(chi_views),
                                                                      std::as_const(spacing_views),
                                                                      grad2_1_1_chi_views);
        auto k_grad2_2_2_chi = KernelZZDeriv3D<KernelHyperparameters>(options,
                                                                      std::as_const(chi_views),
                                                                      std::as_const(spacing_views),
                                                                      grad2_2_2_chi_views);
        auto k_nlsm_rhs      = KernelNLSMRHS<KernelHyperparameters>(options,
                                                               std::as_const(chi_views),
                                                               std::as_const(phi_views),
                                                               std::as_const(grad2_0_0_chi_views),
                                                               std::as_const(grad2_1_1_chi_views),
                                                               std::as_const(grad2_2_2_chi_views),
                                                               chi_rhs_views,
                                                               phi_rhs_views);


        printf("\nbuilding optimizer\n");
        auto kernels = pack(k_grad2_0_0_chi, k_grad2_1_1_chi, k_grad2_2_2_chi, k_nlsm_rhs);

        Optimizer algo(kernels, data_views, reordering);
        algo.set_num_chain_runs(num_chain_runs);


        // TODO: algo validation function
        algo.set_validation_function([&]() {});


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
#endif
    Kokkos::finalize();

    printf("\n** GRACEFUL EXIT **\n");
}
