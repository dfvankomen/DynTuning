#include "optimizer.hpp"

#include "Kokkos_Core.hpp"
#include "common.hpp"
#include "data.hpp"
#include "data_transfers.hpp"
#include "kernel_matvecmult.hpp"
#include "kernel_vectordot.hpp"
#include "test_config.hpp"
#include "view.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cstddef>
#include <utility>

typedef Kokkos::Cuda DeviceExecSpace;
typedef Kokkos::RangePolicy<DeviceExecSpace> device_range_policy;
typedef Kokkos::MDRangePolicy<DeviceExecSpace, Kokkos::Rank<2>> device_rank2_range_policy;

TEST_CASE("Optimizer: Test No Dependencies, No Reordering")
{
    const size_t N        = 10;
    const size_t M        = 20;
    const bool reordering = false;

    // execution options
    KernelOptions options;
    if (GLOBAL_DEVICE != DeviceSelector::AUTO)
        options = { { GLOBAL_DEVICE } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };
    {

#include "helpers/optimizer_build_views.cpp"

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
        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k1                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(a_views),
                                                         std::as_const(b_views),
                                                         c_views);
        auto k2                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(d_views),
                                                         std::as_const(e_views),
                                                         f_views);

        auto kernels = pack(k1, k2);

        Optimizer algo(kernels, data_views, reordering);

        // do some checks on the optimizer's internals
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

        // the optimizer "graph" is the simple graph used for traversal based on index.
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
        if (options.devices.size() == 1)
        {
            // if there's only one device, there's only one chain!
            REQUIRE(algo.kernel_chains.size() == 1);
        }
        else
        {
            // in this case, each of the two can either both be on host, one be on device, or both
            // be on device. that makes 4.
            REQUIRE(algo.kernel_chains.size() == 4);
        }

        // then set up our optimizer verification function to test both of our outputs!
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

        printf("\nNow running optimizer...\n");
        algo();
        printf("\nFinished running optimizer!\n");

        // DONE
    }
}



TEST_CASE("Optimizer: Test No Dependencies, W/ Reordering")
{
    const size_t N        = 10;
    const size_t M        = 20;
    const bool reordering = true;

    // execution options
    KernelOptions options;
    if (GLOBAL_DEVICE != DeviceSelector::AUTO)
        options = { { GLOBAL_DEVICE } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    {

#include "helpers/optimizer_build_views.cpp"

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
        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k1                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(a_views),
                                                         std::as_const(b_views),
                                                         c_views);
        auto k2                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(d_views),
                                                         std::as_const(e_views),
                                                         f_views);

        auto kernels = pack(k1, k2);

        Optimizer algo(kernels, data_views, reordering);

        // do some checks on the optimizer's internals
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

        // the optimizer "graph" is the simple graph used for traversal based on index.
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
        if (options.devices.size() == 1)
        {
            // with only one device, and the data dependencies, we only have 2
            REQUIRE(algo.kernel_chains.size() == 2);
        }
        else
        {
            // in this case, each of the two can either both be on host, one be on device, or both
            // be on device. that makes 8 since they can go in either order.
            REQUIRE(algo.kernel_chains.size() == 8);
        }

        // then set up our optimizer verification function to test both of our outputs!
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

        printf("\nNow running optimizer...\n");
        algo();
        printf("\nFinished running optimizer!\n");

        algo.print_results();
    }
}


TEST_CASE("Optimizer: 3 Kernels, 1 Dependent, No Reordering")
{
    const size_t N        = 10;
    const size_t M        = 20;
    const bool reordering = false;

    // execution options
    KernelOptions options;
    if (GLOBAL_DEVICE != DeviceSelector::AUTO)
        options = { { GLOBAL_DEVICE } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    {

#include "helpers/optimizer_build_views.cpp"

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

        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k1                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(a_views),
                                                         std::as_const(b_views),
                                                         c_views);
        auto k2                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(d_views),
                                                         std::as_const(e_views),
                                                         f_views);
        auto k3                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(c_views),
                                                         std::as_const(g_views),
                                                         h_views);

        auto kernels = pack(k1, k2, k3);

        Optimizer algo(kernels, data_views, reordering);

        // do some checks on the optimizer's internals
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

        // the optimizer "graph" is the simple graph used for traversal based on index.
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
        if (options.devices.size() == 1)
        {
            // if there's only one device, there's only one chain!
            REQUIRE(algo.kernel_chains.size() == 1);
        }
        else
        {
            // in this case, the output of kernel 1 must always match the input of kernel 2, so that
            // reduces the number a bit, so we have 8 chains
            REQUIRE(algo.kernel_chains.size() == 8);
        }

        // then set up our optimizer verification function to test both of our outputs!
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

        printf("\nNow running optimizer...\n");
        algo();
        printf("\nFinished running optimizer!\n");

        // DONE
    }
}

TEST_CASE("Optimizer: 3 Kernels, 1 Dependent, Reordering")
{
    const size_t N        = 10;
    const size_t M        = 20;
    const bool reordering = true;

    // execution options
    KernelOptions options;
    if (GLOBAL_DEVICE != DeviceSelector::AUTO)
        options = { { GLOBAL_DEVICE } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    {

#include "helpers/optimizer_build_views.cpp"

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

        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k1                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(a_views),
                                                         std::as_const(b_views),
                                                         c_views);
        auto k2                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(d_views),
                                                         std::as_const(e_views),
                                                         f_views);
        auto k3                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(c_views),
                                                         std::as_const(g_views),
                                                         h_views);

        auto kernels = pack(k1, k2, k3);

        Optimizer algo(kernels, data_views, reordering);

        // do some checks on the optimizer's internals
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

        // the optimizer "graph" is the simple graph used for traversal based on index.
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
        if (options.devices.size() == 1)
        {
            // because of the data dependencies, we have 3 if we only have one device
            REQUIRE(algo.kernel_chains.size() == 3);
        }
        else
        {
            // in this case, the output of kernel 1 must always match the input of kernel 2, so that
            // reduces the number a bit, so we have 24 chains
            REQUIRE(algo.kernel_chains.size() == 24);
        }

        // then set up our optimizer verification function to test both of our outputs!
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

        printf("\nNow running optimizer...\n");
        algo();
        printf("\nFinished running optimizer!\n");

        // DONE
    }
}



TEST_CASE("Optimizer: Test Multiple Algos with Same Data")
{
    const size_t N        = 10;
    const size_t M        = 20;
    const bool reordering = false;

    // execution options
    KernelOptions options;
    if (GLOBAL_DEVICE != DeviceSelector::AUTO)
        options = { { GLOBAL_DEVICE } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    {

#include "helpers/optimizer_build_views.cpp"

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

        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k1                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(a_views),
                                                         std::as_const(b_views),
                                                         c_views);
        auto k2                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(d_views),
                                                         std::as_const(e_views),
                                                         f_views);
        auto k3                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(c_views),
                                                         std::as_const(g_views),
                                                         h_views);

        auto kernels = pack(k1, k2, k3);

        Optimizer algo(kernels, data_views, reordering);
        Optimizer algo2(kernels, data_views, reordering);

        // do some checks on the optimizer's internals
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

        // the optimizer "graph" is the simple graph used for traversal based on index.
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
        if (options.devices.size() == 1)
        {
            // if there's only one device, there's only one chain!
            REQUIRE(algo.kernel_chains.size() == 1);
        }
        else
        {
            // in this case, the output of kernel 1 must always match the input of kernel 2, so that
            // reduces the number a bit, so we have 8 chains
            REQUIRE(algo.kernel_chains.size() == 8);
        }

        auto val_function = [&f, &h, &f_truth, &h_truth, &N]()
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
        };

        // then set up our optimizer verification function to test both of our outputs!
        algo.set_validation_function(val_function);
        algo2.set_validation_function(val_function);

        printf("\nNow running optimizer...\n");
        algo();
        printf("\nFinished running optimizer!\n");


        printf("\nNow running optimizer 2...\n");
        algo2();
        printf("\nFinished running optimizer!\n");

        // DONE
    }
}



TEST_CASE("Optimizer: Test Multiple Algos (Diff Kernels) with Same Data")
{
    const size_t N        = 10;
    const size_t M        = 20;
    const bool reordering = false;

    // execution options
    KernelOptions options;
    if (GLOBAL_DEVICE != DeviceSelector::AUTO)
        options = { { GLOBAL_DEVICE } };
    else
        options = { { DeviceSelector::HOST, DeviceSelector::DEVICE } };

    {

#include "helpers/optimizer_build_views.cpp"

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

        using ChosenLinspace        = NoOptions;
        using KernelHyperparameters = HyperparameterOptions<ChosenLinspace>;
        auto k1                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(a_views),
                                                         std::as_const(b_views),
                                                         c_views);
        auto k2                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(d_views),
                                                         std::as_const(e_views),
                                                         f_views);
        auto k3                     = KernelVectorDot<KernelHyperparameters>(options,
                                                         std::as_const(c_views),
                                                         std::as_const(g_views),
                                                         h_views);

        auto kernels  = pack(k1, k2, k3);
        auto kernels2 = pack(k1, k2);

        Optimizer algo(kernels, data_views, reordering);
        Optimizer algo2(kernels2, data_views, reordering);

        auto val_function = [&f, &h, &f_truth, &h_truth, &N]()
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
        };

        auto val_function2 = [&c, &f, &c_truth, &f_truth, &N]()
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
        };

        // then set up our optimizer verification function to test both of our outputs!
        algo.set_validation_function(val_function);
        algo2.set_validation_function(val_function2);

        printf("\nNow running optimizer...\n");
        algo();
        printf("\nFinished running optimizer!\n");


        printf("\nNow running optimizer 2...\n");
        algo2();
        printf("\nFinished running optimizer!\n");

        // DONE
    }
}
