#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "kernel.hpp"
#include "range.hpp"
#include "view.hpp"

#include <Eigen/Eigen>

#define NLSM_WAVE_SPEED_X 1.0
#define NLSM_WAVE_SPEED_Y 1.0
#define NLSM_WAVE_SPEED_Z 1.0

struct FunctorKernel_NLSMRHS_Host
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views,
                                    const Index i,
                                    const Index j,
                                    const Index k) const
    {
        auto chi = std::get<0>(views);
        auto phi = std::get<1>(views);
        // derivatives (also inputs)
        auto grad2_0_0_chi = std::get<2>(views);
        auto grad2_1_1_chi = std::get<3>(views);
        auto grad2_2_2_chi = std::get<4>(views);

        // outputs
        auto chi_rhs = std::get<5>(views);
        auto phi_rhs = std::get<6>(views);

        // make sure we're within the boundaries of our padding region
        if (i > 3 && i < chi.extent(0) - 3 && j > 3 && j < chi.extent(1) - 3 && k > 3 &&
            k < chi.extent(2) - 3)
        {
            phi_rhs(i, j, k) = NLSM_WAVE_SPEED_X * grad2_0_0_chi(i, j, k) +
                               NLSM_WAVE_SPEED_Y * grad2_1_1_chi(i, j, k) +
                               NLSM_WAVE_SPEED_Z * grad2_2_2_chi(i, j, k);
            chi_rhs(i, j, k) = phi(i, j, k);
        }
    }
};


struct FunctorKernel_NLSMRHS_Device
{
    template<typename ViewsTuple, typename Index>
    KOKKOS_FUNCTION void operator()(ViewsTuple views,
                                    const Index i,
                                    const Index j,
                                    const Index k) const
    {
#if 0
        auto chi = std::get<0>(views);
        auto phi = std::get<1>(views);
        // derivatives (also inputs)
        auto grad2_0_0_chi = std::get<2>(views);
        auto grad2_1_1_chi = std::get<3>(views);
        auto grad2_2_2_chi = std::get<4>(views);

        // outputs
        auto chi_rhs = std::get<5>(views);
        auto phi_rhs = std::get<6>(views);
#endif


        // const int chi_n = chi.extent(0);
        // const int chi_m = chi.extent(1);
        // const int chi_k = chi.extent(2);
        //
        // const int chi_rhs_n = chi_rhs.extent(0);
        // const int chi_rhs_m = chi_rhs.extent(1);
        // const int chi_rhs_k = chi_rhs.extent(2);
        //
        // const int phi_rhs_n = phi_rhs.extent(0);
        // const int phi_rhs_m = phi_rhs.extent(1);
        // const int phi_rhs_k = phi_rhs.extent(2);



        // printf("chi: %p, phi: %p, chi_rhs: %p, phi_rhs: %p\n",
        //        chi.data(),
        //        phi.data(),
        //        chi_rhs.data(),
        //        phi_rhs.data());



#if 0
        // make sure we're within the boundaries of our padding region
        if (i > 3 && i < chi.extent(0) - 3 && j > 3 && j < chi.extent(1) - 3 && k > 3 &&
            k < chi.extent(2) - 3)
        {
            phi_rhs(i, j, k) = NLSM_WAVE_SPEED_X * grad2_0_0_chi(i, j, k) +
                               NLSM_WAVE_SPEED_Y * grad2_1_1_chi(i, j, k) +
                               NLSM_WAVE_SPEED_Z * grad2_2_2_chi(i, j, k);
            chi_rhs(i, j, k) = phi(i, j, k);
        }
#endif
    }
};


template<typename HyperparameterConfig, typename... ParameterTypes>
inline auto KernelNLSMRHS(KernelOptions& options, ParameterTypes&... data_views)
{
    constexpr int KernelRank = 3;

    auto name = "nlsm rhs kernel";

    auto is_const = kernel_io_map(data_views...);
    auto views_   = pack(data_views...);
    auto views    = repack_views(views_);

    auto out = std::get<1>(std::get<0>(views));

    unsigned long N = out.extent(0);
    unsigned long M = out.extent(1);
    unsigned long K = out.extent(2);

    std::cout << "NMK: " << N << " " << M << " " << K << std::endl;

    RangeExtent<KernelRank> extent = range_extent(Kokkos::Array<std::uint64_t, 3> { 0, 0, 0 },
                                                  Kokkos::Array<std::uint64_t, 3> { N, M, K });

    auto full_policy_collection =
      make_policy_from_hyperparameters<KernelRank, Kokkos::KOKKOS_DEVICE, HyperparameterConfig>(
        extent);
    auto policy_names = make_hyperparameter_vector<HyperparameterConfig>();

    return Kernel<KernelRank,
                  FunctorKernel_NLSMRHS_Host,
                  FunctorKernel_NLSMRHS_Device,
                  decltype(views),
                  decltype(is_const),
                  decltype(full_policy_collection),
                  decltype(policy_names)>(name,
                                          views,
                                          is_const,
                                          extent,
                                          options,
                                          full_policy_collection,
                                          policy_names);
}
