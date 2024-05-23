// Assumptions:
// * Algorithms are cast as operating on one element at a time
// * Kernels are steps in the algorithm that typically involve a loop over the elements
// * Inputs and outputs are all ndarrays at the element level
//

#include "common.hpp"
#include "Kokkos_Core.hpp"

#include "algorithm.hpp"
#include "kernel_vvm.hpp"
#include "kernel_mvm.hpp"
#include "data.hpp"

/*
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
*/

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[])
{
    DeviceSelector device = set_device(argc, argv);
    int                 N = set_N(argc, argv);
    bool       reordering = set_reordering(argc, argv);
    bool       initialize = set_initialize(argc, argv);
    
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
        std::vector<double> q(N);       
        std::vector<double> r(N);
        std::vector<double> s(N);
        //Eigen::MatrixXd  t(N, N);
        std::vector<double> u(N);
        std::vector<double> v(N);
        std::vector<double> w(N);
        std::vector<double> x(N);
        std::vector<double> y(N);
        std::vector<double> z(N);

        // initialize data
        if (initialize) {
            printf("\ninitializing data\n");
            std::iota(q.begin(), q.end(), 1.0);
            std::iota(r.begin(), r.end(), 1.0);
            std::iota(s.begin(), s.end(), 1.0);
            //{ int ij = 0; for (int i=0; i<N; i++) for (int j=0; j<N; j++) t(i,j) = static_cast<double>(ij++); }
            std::iota(v.begin(), v.end(), 1.0);
            std::iota(u.begin(), u.end(), 1.0); // temporary: initialized so we can disable the 2D cases
            std::iota(x.begin(), x.end(), 1.0);
            std::iota(y.begin(), y.end(), 1.0);
        }

        // register all data into a DataManager
        // Has to be constexpr to use later in a constexpr invocation of find()
        // I have not found a cleaner way to write this...
        // I feel like some kind of recursive function or macro combo should work...
        constexpr auto data_names = std::make_tuple(
            HashedName<hash("q")>(),
            HashedName<hash("r")>(),
            HashedName<hash("s")>(),
            //HashedName<hash("t")>(),
            HashedName<hash("u")>(),
            HashedName<hash("v")>(),
            HashedName<hash("w")>(),
            HashedName<hash("x")>(),
            HashedName<hash("y")>(),
            HashedName<hash("z")>()
        );
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
        auto data_views = create_views(pack(q, r, s, u, v, w, x, y, z));

        auto& q_views = std::get<find<hash("q")>(data_names)>(data_views);
        auto& r_views = std::get<find<hash("r")>(data_names)>(data_views);
        auto& s_views = std::get<find<hash("s")>(data_names)>(data_views);
        //auto& t_views = std::get<find<hash("t")>(data_names)>(data_views);
        auto& u_views = std::get<find<hash("u")>(data_names)>(data_views);
        auto& v_views = std::get<find<hash("v")>(data_names)>(data_views);
        auto& w_views = std::get<find<hash("w")>(data_names)>(data_views);
        auto& x_views = std::get<find<hash("x")>(data_names)>(data_views);
        auto& y_views = std::get<find<hash("y")>(data_names)>(data_views);
        auto& z_views = std::get<find<hash("z")>(data_names)>(data_views);

        // define all kernels
        printf("\nbuilding kernels\n");
        auto k1 = KernelVVM(options, std::as_const(q_views), std::as_const(r_views), s_views); // vvm
        //auto k2 = KernelMVM(options, std::as_const(t_views), std::as_const(s_views), u_views); // mvm
        auto k3 = KernelVVM(options, std::as_const(v_views), std::as_const(u_views), w_views); // vvm
        //auto k5 = KernelVVM(options, std::as_const(v_views), std::as_const(u_views), w_views); // vvm
        auto k4 = KernelVVM(options, std::as_const(s_views), std::as_const(y_views), z_views); // vvm
       


        // register all kernels info an Algorithm
        //auto kernels = pack(k1, k2, k3, k4);
        printf("\nbuilding algorithm\n");
        auto kernels = pack(k1, k3, k4);
        Algorithm algo(kernels, data_views, reordering);

        // run the algorithm
        printf("\nrunning algorithm\n");
        algo();

        // TESTS
        //TestVVM(k1, q, r, s);
        //TestMVM(k2, t, s, u);
        //TestVVM(k3, v, u, w);
        //TestVVM(k4, x, y, z);

    } // end Kokkos scope
    Kokkos::finalize();

    printf("\n** GRACEFUL EXIT **\n");
}
