#include "algorithm.hpp"

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"
#include "data_transfers.hpp"
#include "kernel_matvecmult.hpp"
#include "kernel_vectordot.hpp"
#include "view.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <utility>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;
typedef Kokkos::MDRangePolicy<DeviceExecSpace, Kokkos::Rank<2>> device_rank2_range_policy;

TEST_CASE("Algorithm: Test No Dependencies, No Reordering")
{
    const size_t N        = 10;
    const size_t M        = 20;
    const bool reordering = false;

    const DeviceSelector device = DeviceSelector::AUTO;

    // execution options
    KernelOptions options;
    if (device != DeviceSelector::AUTO)
        options = { { device } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    Kokkos::initialize();
    {

#include "helpers/algorithm_build_views.cpp"

        // truth vectors for numeric computations
        std::vector<double> c_truth(N);
        std::vector<double> f_truth(N);

        for (size_t i = 0; i < N; i++)
        {
            c_truth[i] = a[i] * b[i];
            f_truth[i] = d[i] * e[i];
            c[i]       = 0.0;
            f[i]       = 0.0;
        }

        // first test is just a few VectorDot kernels that should work
        auto k1 = KernelVectorDot(options, std::as_const(a_views), std::as_const(b_views), c_views);
        auto k2 = KernelVectorDot(options, std::as_const(d_views), std::as_const(e_views), f_views);

        auto kernels = pack(k1, k2);

        Algorithm algo(kernels, data_views, reordering);

        // do some checks on the algorithm's internals
        REQUIRE(algo.reordering_ == reordering);

        // check the size of detected inputs and outputs
        REQUIRE(algo.inputs.size() == 4);
        REQUIRE(algo.outputs.size() == 2);

        // dependents is a vector of sets, for kernel set, we have nothing
        for (auto& dep : algo.dependents)
        {
            REQUIRE(dep.size() == 0);
        }
        // depends_on is similar, it's the "prev" in a directional graph
        for (auto& dep : algo.depends_on)
        {
            REQUIRE(dep.size() == 0);
        }

        // the algorithm "graph" is the simple graph used for traversal based on index.
        // this index is the kernel ID, which is the order passed into the tuple.
        // In particular, this is the KERNEL version of the dependencies
        for (auto& data_node : algo.graph)
        {
            REQUIRE(data_node.prev.size() == 0);
            REQUIRE(data_node.next.size() == 0);
        }

        // the data dependecy graph is a bit trickier to verify, since it stores everything
        for (auto& data_dep : algo.data_graph.graph)
        {
            const size_t view_id = data_dep.first;
            // views a, b, d, e (inputs)
            if (view_id == 2 || view_id == 3 || view_id == 5 || view_id == 6)
            {
                REQUIRE(data_dep.second.prev.size() == 0);
                REQUIRE(data_dep.second.next.size() == 1);
            }
            // views c and f (outputs)
            else if (view_id == 4 || view_id == 7)
            {
                REQUIRE(data_dep.second.prev.size() == 2);
                REQUIRE(data_dep.second.next.size() == 0);
            }
        }

        // also check the number of chains!
        // in this case, each of the two can either both be on host, one be on device, or both be on
        // device. that makes 4.
        REQUIRE(algo.kernel_chains.size() == 4);

        // then set up our algorithm verification function to test both of our outputs!
        algo.set_validation_function([&a, &b, &c, &d, &e, &f, &c_truth, &f_truth, &N]()
        {
            double abs_difference = 0.0;
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(c[i] - c_truth[i]);
                c[i] = 0.0;
            };
            REQUIRE_THAT(abs_difference, Catch::Matchers::WithinRel(0.0, 1.0e-10));

            abs_difference = 0.0;
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(f[i] - f_truth[i]);
                f[i] = 0.0;
            };
            REQUIRE_THAT(abs_difference, Catch::Matchers::WithinRel(0.0, 1.0e-10));
        });

        printf("\nNow running algorithm...\n");
        algo();
        printf("\nFinished running algorithm!\n");

        // DONE
    }
    Kokkos::finalize();
}



TEST_CASE("Algorithm: Test No Dependencies, W/ Reordering")
{
    const size_t N        = 10;
    const size_t M        = 20;
    const bool reordering = true;

    const DeviceSelector device = DeviceSelector::AUTO;

    // execution options
    KernelOptions options;
    if (device != DeviceSelector::AUTO)
        options = { { device } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    Kokkos::initialize();
    {

#include "helpers/algorithm_build_views.cpp"

        // truth vectors for numeric computations
        std::vector<double> c_truth(N);
        std::vector<double> f_truth(N);

        for (size_t i = 0; i < N; i++)
        {
            c_truth[i] = a[i] * b[i];
            f_truth[i] = d[i] * e[i];
            c[i]       = 0.0;
            f[i]       = 0.0;
        }

        // first test is just a few VectorDot kernels that should work
        auto k1 = KernelVectorDot(options, std::as_const(a_views), std::as_const(b_views), c_views);
        auto k2 = KernelVectorDot(options, std::as_const(d_views), std::as_const(e_views), f_views);

        auto kernels = pack(k1, k2);

        Algorithm algo(kernels, data_views, reordering);

        // do some checks on the algorithm's internals
        REQUIRE(algo.reordering_ == reordering);

        // check the size of detected inputs and outputs
        REQUIRE(algo.inputs.size() == 4);
        REQUIRE(algo.outputs.size() == 2);

        // dependents is a vector of sets, for kernel set, we have nothing
        for (auto& dep : algo.dependents)
        {
            REQUIRE(dep.size() == 0);
        }
        // depends_on is similar, it's the "prev" in a directional graph
        for (auto& dep : algo.depends_on)
        {
            REQUIRE(dep.size() == 0);
        }

        // the algorithm "graph" is the simple graph used for traversal based on index.
        // this index is the kernel ID, which is the order passed into the tuple.
        // In particular, this is the KERNEL version of the dependencies
        for (auto& data_node : algo.graph)
        {
            REQUIRE(data_node.prev.size() == 0);
            REQUIRE(data_node.next.size() == 0);
        }

        // the data dependecy graph is a bit trickier to verify, since it stores everything
        for (auto& data_dep : algo.data_graph.graph)
        {
            const size_t view_id = data_dep.first;
            // views a, b, d, e (inputs)
            if (view_id == 2 || view_id == 3 || view_id == 5 || view_id == 6)
            {
                REQUIRE(data_dep.second.prev.size() == 0);
                REQUIRE(data_dep.second.next.size() == 1);
            }
            // views c and f (outputs)
            else if (view_id == 4 || view_id == 7)
            {
                REQUIRE(data_dep.second.prev.size() == 2);
                REQUIRE(data_dep.second.next.size() == 0);
            }
        }

        // also check the number of chains!
        // in this case, each of the two can either both be on host, one be on device, or both be on
        // device. that makes 8 since they can go in either order.
        REQUIRE(algo.kernel_chains.size() == 8);

        // then set up our algorithm verification function to test both of our outputs!
        algo.set_validation_function([&c, &f, &c_truth, &f_truth, &N]()
        {
            double abs_difference = 0.0;
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(c[i] - c_truth[i]);
                c[i] = 0.0;
            };
            REQUIRE_THAT(abs_difference, Catch::Matchers::WithinRel(0.0, 1.0e-10));

            abs_difference = 0.0;
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(f[i] - f_truth[i]);
                f[i] = 0.0;
            };
            REQUIRE_THAT(abs_difference, Catch::Matchers::WithinRel(0.0, 1.0e-10));
        });

        printf("\nNow running algorithm...\n");
        algo();
        printf("\nFinished running algorithm!\n");

        algo.print_results();
    }
    Kokkos::finalize();
}


TEST_CASE("Algorithm: 3 Kernels, 1 Dependent, No Reordering")
{
    const size_t N        = 10;
    const size_t M        = 20;
    const bool reordering = false;

    const DeviceSelector device = DeviceSelector::AUTO;

    // execution options
    KernelOptions options;
    if (device != DeviceSelector::AUTO)
        options = { { device } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    Kokkos::initialize();
    {

#include "helpers/algorithm_build_views.cpp"

        // truth vectors for numeric computations
        std::vector<double> c_truth(N);
        std::vector<double> f_truth(N);
        std::vector<double> h_truth(N);

        for (size_t i = 0; i < N; i++)
        {
            c_truth[i] = a[i] * b[i];
            c[i]       = 0.0;
            f_truth[i] = d[i] * e[i];
            f[i]       = 0.0;
        }

        for (size_t i = 0; i < N; i++)
        {
            h_truth[i] = c_truth[i] * g[i];
            h[i]       = 0.0;
        }

        // this is basically kernels 1, 3, and 4 from before

        auto k1 = KernelVectorDot(options, std::as_const(a_views), std::as_const(b_views), c_views);
        auto k2 = KernelVectorDot(options, std::as_const(d_views), std::as_const(e_views), f_views);
        auto k3 = KernelVectorDot(options, std::as_const(c_views), std::as_const(g_views), h_views);

        auto kernels = pack(k1, k2, k3);

        Algorithm algo(kernels, data_views, reordering);

        // do some checks on the algorithm's internals
        REQUIRE(algo.reordering_ == reordering);

        // check the size of detected inputs and outputs
        REQUIRE(algo.inputs.size() == 6);
        REQUIRE(algo.outputs.size() == 3);

        // dependents is a vector of sets, for this kernel we have different values depending on the
        // kernel
        for (size_t i = 0; i < algo.dependents.size(); i++)
        {
            auto& dep = algo.dependents[i];
            if (i == 0)
            {
                REQUIRE(dep.size() == 1);
            }
            else
            {
                REQUIRE(dep.size() == 0);
            }
        }
        // depends_on is similar, it's the "prev" in a directional graph
        for (size_t i = 0; i < algo.depends_on.size(); i++)
        {
            auto& dep_on = algo.depends_on[i];
            if (i == 2)
            {
                REQUIRE(dep_on.size() == 1);
            }
            else
            {
                REQUIRE(dep_on.size() == 0);
            }
        }

        // the algorithm "graph" is the simple graph used for traversal based on index.
        // this index is the kernel ID, which is the order passed into the tuple.
        // In particular, this is the KERNEL version of the dependencies
        for (size_t i = 0; i < algo.graph.size(); i++)
        {
            auto& data_node = algo.graph[i];

            // prev node
            if (i == 2)
            {
                REQUIRE(data_node.prev.size() == 1);
            }
            else
            {
                REQUIRE(data_node.prev.size() == 0);
            }

            if (i == 0)
            {
                REQUIRE(data_node.next.size() == 1);
            }
            else
            {
                REQUIRE(data_node.next.size() == 0);
            }
        }

        // the data dependecy graph is a bit trickier to verify, since it stores everything
        for (auto& data_dep : algo.data_graph.graph)
        {
            const size_t view_id = data_dep.first;

            if (view_id == 4 || view_id == 7 || view_id == 9)
                REQUIRE(data_dep.second.prev.size() == 2);
            else
                REQUIRE(data_dep.second.prev.size() == 0);

            if (view_id == 8 || view_id == 6 || view_id == 5 || view_id == 3 || view_id == 4 ||
                view_id == 2)
                REQUIRE(data_dep.second.next.size() == 1);
            else
                REQUIRE(data_dep.second.next.size() == 0);
        }

        // also check the number of chains!
        // in this case, the output of kernel 1 must always match the input of kernel 2, so that
        // reduces the number a bit, so we have 8 chains
        REQUIRE(algo.kernel_chains.size() == 8);

        // then set up our algorithm verification function to test both of our outputs!
        algo.set_validation_function([&f, &h, &f_truth, &h_truth, &N]()
        {
            double abs_difference = 0.0;
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(f[i] - f_truth[i]);
                f[i] = 0.0;
            };
            REQUIRE_THAT(abs_difference, Catch::Matchers::WithinRel(0.0, 1.0e-10));

            abs_difference = 0.0;
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(h[i] - h_truth[i]);
                h[i] = 0.0;
            };
            REQUIRE_THAT(abs_difference, Catch::Matchers::WithinRel(0.0, 1.0e-10));
        });

        printf("\nNow running algorithm...\n");
        algo();
        printf("\nFinished running algorithm!\n");

        // DONE
    }
    Kokkos::finalize();
}

TEST_CASE("Algorithm: 3 Kernels, 1 Dependent, Reordering")
{
    const size_t N        = 10;
    const size_t M        = 20;
    const bool reordering = true;

    const DeviceSelector device = DeviceSelector::AUTO;

    // execution options
    KernelOptions options;
    if (device != DeviceSelector::AUTO)
        options = { { device } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    Kokkos::initialize();
    {

#include "helpers/algorithm_build_views.cpp"

        // truth vectors for numeric computations
        std::vector<double> c_truth(N);
        std::vector<double> f_truth(N);
        std::vector<double> h_truth(N);

        for (size_t i = 0; i < N; i++)
        {
            c_truth[i] = a[i] * b[i];
            c[i]       = 0.0;
            f_truth[i] = d[i] * e[i];
            f[i]       = 0.0;
        }

        for (size_t i = 0; i < N; i++)
        {
            h_truth[i] = c_truth[i] * g[i];
            h[i]       = 0.0;
        }

        // this is basically kernels 1, 3, and 4 from before

        auto k1 = KernelVectorDot(options, std::as_const(a_views), std::as_const(b_views), c_views);
        auto k2 = KernelVectorDot(options, std::as_const(d_views), std::as_const(e_views), f_views);
        auto k3 = KernelVectorDot(options, std::as_const(c_views), std::as_const(g_views), h_views);

        auto kernels = pack(k1, k2, k3);

        Algorithm algo(kernels, data_views, reordering);

        // do some checks on the algorithm's internals
        REQUIRE(algo.reordering_ == reordering);

        // check the size of detected inputs and outputs
        REQUIRE(algo.inputs.size() == 6);
        REQUIRE(algo.outputs.size() == 3);

        // dependents is a vector of sets, for this kernel we have different values depending on the
        // kernel
        for (size_t i = 0; i < algo.dependents.size(); i++)
        {
            auto& dep = algo.dependents[i];
            if (i == 0)
            {
                REQUIRE(dep.size() == 1);
            }
            else
            {
                REQUIRE(dep.size() == 0);
            }
        }
        // depends_on is similar, it's the "prev" in a directional graph
        for (size_t i = 0; i < algo.depends_on.size(); i++)
        {
            auto& dep_on = algo.depends_on[i];
            if (i == 2)
            {
                REQUIRE(dep_on.size() == 1);
            }
            else
            {
                REQUIRE(dep_on.size() == 0);
            }
        }

        // the algorithm "graph" is the simple graph used for traversal based on index.
        // this index is the kernel ID, which is the order passed into the tuple.
        // In particular, this is the KERNEL version of the dependencies
        for (size_t i = 0; i < algo.graph.size(); i++)
        {
            auto& data_node = algo.graph[i];

            // prev node
            if (i == 2)
            {
                REQUIRE(data_node.prev.size() == 1);
            }
            else
            {
                REQUIRE(data_node.prev.size() == 0);
            }

            if (i == 0)
            {
                REQUIRE(data_node.next.size() == 1);
            }
            else
            {
                REQUIRE(data_node.next.size() == 0);
            }
        }

        // the data dependecy graph is a bit trickier to verify, since it stores everything
        for (auto& data_dep : algo.data_graph.graph)
        {
            const size_t view_id = data_dep.first;

            if (view_id == 4 || view_id == 7 || view_id == 9)
                REQUIRE(data_dep.second.prev.size() == 2);
            else
                REQUIRE(data_dep.second.prev.size() == 0);

            if (view_id == 8 || view_id == 6 || view_id == 5 || view_id == 3 || view_id == 4 ||
                view_id == 2)
                REQUIRE(data_dep.second.next.size() == 1);
            else
                REQUIRE(data_dep.second.next.size() == 0);
        }

        // also check the number of chains!
        // in this case, the output of kernel 1 must always match the input of kernel 2, so that
        // reduces the number a bit, so we have 24 chains
        REQUIRE(algo.kernel_chains.size() == 24);

        // then set up our algorithm verification function to test both of our outputs!
        algo.set_validation_function([&f, &h, &f_truth, &h_truth, &N]()
        {
            double abs_difference = 0.0;
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(f[i] - f_truth[i]);
                f[i] = 0.0;
            };
            REQUIRE_THAT(abs_difference, Catch::Matchers::WithinRel(0.0, 1.0e-10));

            abs_difference = 0.0;
            for (size_t i = 0; i < N; i++)
            {
                abs_difference += std::abs(h[i] - h_truth[i]);
                h[i] = 0.0;
            };
            REQUIRE_THAT(abs_difference, Catch::Matchers::WithinRel(0.0, 1.0e-10));
        });

        printf("\nNow running algorithm...\n");
        algo();
        printf("\nFinished running algorithm!\n");

        // DONE
    }
    Kokkos::finalize();
}