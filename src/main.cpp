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

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[])
{
    DeviceSelector device = set_device(argc, argv);

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    { // start Kokkos scope

        int N;

        N = 5;
        std::vector<double> x(N);
        std::iota(x.begin(), x.end(), 1.0);
        std::vector<double> y(N);
        std::iota(y.begin(), y.end(), 1.0);
        std::vector<double> z(N);
        std::vector<double> w(N);
        std::vector<double> q(N);

        auto k1 = KernelVVM(std::as_const(x), std::as_const(y), z); // vvm
        auto k2 = KernelVVM(std::as_const(x), std::as_const(z), w); // vvm
        auto k3 = KernelVVM(std::as_const(x), std::as_const(z), q); // vvm

        #ifdef USE_EIGEN
            Eigen::MatrixXd a(N, N);
            //a.setIdentity();
            /*{
            int ij = 0;
            for (int i=0; i<N; i++)
                for (int j=0; j<N; j++)
                a(i,j) = static_cast<double>(ij++);
            }*/
            a << 0.0, 1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0, 9.0,
                10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0, 17.0, 18.0, 19.0,
                20.0, 21.0, 22.0, 23.0, 24.0;
            std::vector<double> b(N, 2.0);
            std::vector<double> c(N, 0.0);

            auto k4 = KernelMVM(std::as_const(a), std::as_const(b), c); // mvm
        #endif

        // Create an Algorithm object
        #ifdef USE_EIGEN
            Algorithm algo(pack(k1, k2, k3, k4));
        #else
            Algorithm algo(pack(k1, k2, k3));
        #endif
        
        // TESTS
        TestVVM(k1, x, y, z, device);
        TestVVM(k2, x, z, w, device);
        TestVVM(k3, x, z, q, device);
        #ifdef USE_EIGEN
            TestMVM(k4, a, b, c, device);
        #endif

    } // end Kokkos scope
    Kokkos::finalize();

    printf("\n** GRACEFUL EXIT **\n");
}
