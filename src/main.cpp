// Assumptions:
// * Algorithms are cast as operating on one element at a time
// * Kernels are steps in the algorithm that typically involve a loop over the elements
// * Inputs and outputs are all ndarrays at the element level
//

// options
// #define USE_EIGEN
// #define NDEBUG

#include "common.hpp"
#include "Kokkos_Core.hpp"

#include "algorithm.hpp"
#include "kernel_vvm.hpp"
#include "kernel_mvm.hpp"
#include "data.hpp"

#define init_data_views(...) \
    auto data_views = create_views(pack(__VA_ARGS__));

#define get_data_view_macro(_1,_2,NAME,...) NAME
#define get_data_view_1(x) \
    std::get<find<hash(x)>(data_names)>(data_views)
#define get_data_view_2(x, device) \
    std::get<device>(std::get<find<hash(x)>(data_names)>(data_views))
#define get_data_view(...) get_data_view_macro(__VA_ARGS__, get_data_view_2, get_data_view_1)(__VA_ARGS__)
#define get_data_view_host(x) get_data_view(x, 0)
#define get_data_view_device(x) get_data_view(x, 1)
#define get_data_view_scratch(x) get_data_view(x, 2)

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[])
{
    DeviceSelector device = set_device(argc, argv);
    int N = set_N(argc, argv);
    bool reordering = set_reordering(argc, argv);
    
    // execution options
    KernelOptions options;
    if (device != DeviceSelector::AUTO)
        options = { { device } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    // Initialize Kokkos
    Kokkos::initialize();
    { // start Kokkos scope
        
        // first define all data
        std::vector<double> x(N);
        std::vector<double> y(N);
        std::vector<double> z(N);
        std::vector<double> w(N);
        std::vector<double> q(N);
        std::vector<double> a(N);
        //#ifdef USE_EIGEN
        //Eigen::MatrixXd a(N, N);
        ////a.setIdentity();
        ///*{
        //int ij = 0;
        //for (int i=0; i<N; i++)
        //    for (int j=0; j<N; j++)
        //    a(i,j) = static_cast<double>(ij++);
        //}*/
        //a << 0.0, 1.0, 2.0, 3.0, 4.0,
        //    5.0, 6.0, 7.0, 8.0, 9.0,
        //    10.0, 11.0, 12.0, 13.0, 14.0,
        //    15.0, 16.0, 17.0, 18.0, 19.0,
        //    20.0, 21.0, 22.0, 23.0, 24.0;
        //std::vector<double> b(N, 2.0);
        //std::vector<double> c(N, 0.0);
        //#endif

        // initialize data
        std::iota(x.begin(), x.end(), 1.0);
        std::iota(y.begin(), y.end(), 1.0);

        // register all data into a DataManager
        // Has to be constexpr to use later in a constexpr invocation of find()
        // I cannot find a cleaner way to write this
        constexpr auto data_names = std::make_tuple(
            HashedName<hash("x")>(),
            HashedName<hash("y")>(),
            HashedName<hash("z")>(),
            HashedName<hash("w")>(),
            HashedName<hash("q")>(),
            HashedName<hash("a")>()
        );
        init_data_views(x, y, z, w, q, a);
        auto& x_views = get_data_view("x");
        auto& y_views = get_data_view("y");
        auto& z_views = get_data_view("z");
        auto& w_views = get_data_view("w");
        auto& q_views = get_data_view("q");
        auto& a_views = get_data_view("a");

        // define all kernels
        auto k1 = KernelVVM(options, std::as_const(x_views), std::as_const(y_views), z_views); // vvm
        auto k2 = KernelVVM(options, std::as_const(x_views), std::as_const(z_views), w_views); // vvm
        auto k3 = KernelVVM(options, std::as_const(x_views), std::as_const(z_views), q_views); // vvm
        auto k4 = KernelVVM(options, std::as_const(x_views), std::as_const(y_views), a_views); // vvm
        //#ifdef USE_EIGEN
        //auto k4 = KernelMVM(std::as_const(a), std::as_const(b), c); // mvm
        //#endif

        // register all kernels info an Algorithm
        //#ifdef USE_EIGEN
        //Algorithm algo(pack(k1, k2, k3, k4), reordering);
        //#else
        auto kernels = pack(k1, k2, k3, k4);
        Algorithm algo(kernels, data_views, reordering);
        //#endif

        // run the algorithm
        algo();

        // TESTS
//        TestVVM(k1, x, y, z);
//        TestVVM(k2, x, z, w);
//        TestVVM(k3, x, z, q);
        //#ifdef USE_EIGEN
        //TestMVM(k4);
        //#endif

    } // end Kokkos scope
    Kokkos::finalize();

    printf("\n** GRACEFUL EXIT **\n");
}
