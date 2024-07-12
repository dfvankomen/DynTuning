#pragma once

#include "common.hpp"
#include "data_deps.hpp"
#include "data_transfers.hpp"
#include "kernel.hpp"

#include <functional>
#include <iomanip>
#include <limits>
#include <set>
#include <stdexcept>

#define DEBUG_ENABLED 0

// main algorithm object
template<typename KernelsTuple, typename ViewsTuple>
class Algorithm
{
  public:
    // constructor should initialize and empty vector
    constexpr Algorithm(KernelsTuple kernels, ViewsTuple views, bool reordering = false)
      : kernels_(kernels)
      , views_(views)
      , reordering_(reordering)
    {
        iter_tuple(kernels_,
                   [&]<typename KernelType>(size_t i, KernelType& k)
        {
            depends_on.push_back(std::set<size_t> {});
            dependents.push_back(std::set<size_t> {});
        });

        build_graph();
        build_expanded_chains();
    };
    ~Algorithm() {};

    // the core of this class is a tuple of kernels
    KernelsTuple kernels_;

    // algorithm is reponsible for shuttling data
    ViewsTuple views_;

    // allow reordering
    bool reordering_;

    // data dependency graph
    DataDependencyGraph data_graph;

    // available options for devices to run on
    std::vector<std::vector<DeviceSelector>> devices;

    // index maps define data dependencies
    using index_pair    = std::tuple<size_t, size_t>;
    using index_map     = std::map<index_pair, index_pair>;
    using adjacency_map = std::vector<std::set<size_t>>;
    index_map inputs;
    index_map outputs;
    adjacency_map depends_on;
    adjacency_map dependents;

    /*
    // call all kernels
    void call()
    {
        iter_tuple(kernels_,
                   []<typename KernelType>(size_t i, KernelType& kernel)
                   { TIMING(kernel, kernel.call()); });
    };
    */

    void set_num_chain_runs(int in)
    {
        chain_runs = in;
    }


#ifdef DYNTUNE_SINGLE_CHAIN_RUN
    void set_selected_chain(unsigned int in)
    {
        selected_only_run = in;
    }
#endif

  private:
    enum
    {
        UNVISITED,
        INPROGRESS,
        VISITED
    };

    // simple index-based DAG node
    struct Node
    {
        size_t i;
        std::vector<size_t> prev;
        std::vector<size_t> next;

        int indegree;
        int state = UNVISITED;

        void reset()
        {
            indegree = prev.size();
            state    = UNVISITED;
        }
    };

    // Function to compute the Cartesian product of arrays
    template<typename T>
    std::vector<std::vector<T>> cartesian_product(const std::vector<std::vector<T>>& arrays)
    {
        std::vector<size_t> indices(arrays.size(), 0);
        bool changed;
        std::vector<std::vector<T>> result;

        do
        {
            std::vector<T> tuple;
            for (size_t i = 0; i < arrays.size(); ++i)
            {
                tuple.push_back(arrays[i][indices[i]]);
            }
            result.push_back(tuple);

            // Update indices
            changed = false;
            for (size_t i = arrays.size(); i-- > 0;)
            {
                ++indices[i];
                if (indices[i] < arrays[i].size())
                {
                    changed = true;
                    break;
                }
                else
                {
                    indices[i] = 0;
                }
            }
        } while (changed);

        return result;
    }

    template<typename T, typename IndexType>
    std::vector<std::vector<T>> reorder_vector(std::vector<std::vector<T>>& B,
                                               std::vector<IndexType>& A)
    {
        std::vector<std::vector<T>> result(B.size());
        for (IndexType i = 0; i < A.size(); ++i)
            result[i] = B[A[i]];
        return result;
    }

    std::function<void()> validation_function = []() {
    };

    struct KernelSelector
    {
        // kernel
        size_t kernel_id;             // index of the kernel in the original tuple
        DeviceSelector kernel_device; // device this kernel will run on
        size_t kernel_n_options; // the number of options in the kernel (currently corresponds to
                                 // Kokkos LaunchBounds)

        // data
        std::vector<size_t> input_id              = std::vector<size_t>();
        std::vector<size_t> output_id             = std::vector<size_t>();
        std::vector<DeviceSelector> output_device = std::vector<DeviceSelector>();
        std::vector<size_t> output_id_local       = std::vector<size_t>();
        std::vector<size_t> input_id_local        = std::vector<size_t>();
    };

    std::vector<uint32_t> chains_total_input_transfers_d2h;
    std::vector<uint32_t> chains_total_output_transfers_d2h;
    std::vector<uint32_t> chains_total_input_transfers_h2d;
    std::vector<uint32_t> chains_total_output_transfers_h2d;

    struct KernelWithOptionsSelector
    {
        size_t original_chain_kd;
        size_t kernel_option_id;
        std::vector<size_t> kernel_option_ids = std::vector<size_t>();
    };

  public:
    // data dependencies define kernel dependencies
    std::vector<Node> graph;

    // valid sequences of kernels, assuming they must run one-at-a-time
    // outer index yields a valid sequence
    // inner index yields a kernel in that sequence
    std::vector<std::vector<size_t>> kernel_sequences;

    // valid concurrent streams of kernels
    // different streams could be scheduled at the same time
    // kernels within a stream must execute sequentially
    // outer index yields an indpendent stream
    // inner index yields a kernel in that stream
    std::vector<std::vector<size_t>> kernel_streams;

    // these are the queues of operations the algorithm will actually execute
    // NOTE right now we always assume kernels are executed in a sequence
    //   however, in the future this could be expanded to allow concurrency of independent subchains
    std::vector<std::vector<KernelSelector>> kernel_chains;
    // to go hand and hand with chains, we need to know which
    std::vector<std::size_t> kernel_chain_device_vals;

    using PermutationStorageType = std::vector<std::vector<std::size_t>>;
    std::vector<PermutationStorageType> kernel_chain_device_permutations;
    std::size_t total_num_permutations = 0;
    std::vector<std::size_t> kernel_chain_device_permutations_bounds;

    // Storage vectors for our profiling results. Note that they're all resized to the proper chain
    // lengths
    std::vector<double> chain_times         = std::vector<double>();
    std::vector<double> chain_elapsed_times = std::vector<double>();
    // the stored randomized order of chains
    std::vector<unsigned int> kernel_chain_ids;

    int total_operations_run = 0;
    int chain_runs           = 1;

    void set_validation_function(std::function<void()> infunc)
    {
        validation_function = infunc;
    }

#ifdef DYNTUNE_SINGLE_CHAIN_RUN
    unsigned int selected_only_run = 0;
#endif

  private:
    // topological search to ensure the graph is acyclic
    void top_search()
    {
        for (auto& node : graph)
            node.reset();
        std::vector<size_t> kseq;
        top_search_impl(kseq);
    }
    void top_search_impl(std::vector<size_t>& kseq)
    {
        for (auto& node : graph)
        {
            if ((node.indegree == 0) && (node.state == UNVISITED))
            {
                node.state = INPROGRESS;
                for (size_t i : node.next)
                {
                    Node& child = graph[i];

                    // NOTE: this doesn't work as expected, so we comment it out for now
                    /*
                    // check for circular dependency
                    if (child.state != UNVISITED)
                        throw std::runtime_error("Circular dependency detected!");
                    */

                    // record edge as visited
                    child.indegree--;
                }

                // append to result
                kseq.push_back(node.i);
                node.state = VISITED;

                // recurse
                top_search_impl(kseq);
                if ((!reordering_) && (kernel_sequences.size() == 1))
                    return;

                // store result
                if (kseq.size() == graph.size())
                {
                    kernel_sequences.push_back(kseq);

                    // if no reordering, then just need first sequence
                    if (!reordering_)
                        return;
                }

                // backtrack
                kseq.erase(kseq.end() - 1);
                node.state = UNVISITED;
                for (size_t i : node.next)
                {
                    Node& child = graph[i];
                    child.indegree++;
                }

            } // end node
        } // end loop over nodes
    }

    // NOTE
    //  the iter_tuple construct is a workaround to access tuple elements with a runtime index
    //  the variant idiom has similar limitations, see
    // https://stackoverflow.com/questions/52088928/trying-to-return-the-value-from-stdvariant-using-stdvisit-and-a-lambda-expre

    // build the DataGraph from chain of Kernels
    void build_graph()
    {
        size_t null_v = std::numeric_limits<std::size_t>::max();

        // 0: loop over left kernel
        iter_tuple(kernels_,
                   [&]<typename KernelTypeL>(size_t il, KernelTypeL& kernel_l) { // kernel_l
            // 1: loop over left data views
            iter_tuple(std::get<0>(kernel_l.data_views_),
                       [&]<typename ViewTypeL>(size_t jl, ViewTypeL& view_l) { // view_l
                // 2: check is_const for jlth data view
                find_tuple(kernel_l.is_const_,
                           jl,
                           [&](bool is_const_l) { // is_const_l
                    // bool is_const_l = false;

                    bool is_match = false;

                    // 3: loop over right kernels
                    iter_tuple(kernels_,
                               [&]<typename KernelTypeJ>(size_t ir,
                                                         KernelTypeJ& kernel_r) { // kernel_r
                        // only look at kernels to the right
                        if (ir <= il)
                            return;

                        // 4: loop over right data views
                        iter_tuple(std::get<0>(kernel_r.data_views_),
                                   [&]<typename ViewTypeR>(size_t jr,
                                                           ViewTypeR& view_r) { // view_r
                            // 5: check is_const for jrth data view
                            find_tuple(kernel_r.is_const_,
                                       jr,
                                       [&](bool is_const_r) { // is_const_r
                                // bool is_const_r = false;

                                // check if there is a data dependency between
                                // these views
                                if ((view_l.data() == view_r.data()) && (!is_const_l) &&
                                    (is_const_r))
                                {
                                    // view_r depends on view_l
                                    outputs.emplace(std::make_tuple(il, jl),
                                                    std::make_tuple(ir, jr));
                                    inputs.emplace(std::make_tuple(ir, jr),
                                                   std::make_tuple(il, jl));

                                    // kernel_r depends on kernel_l
                                    depends_on[ir].insert(il);
                                    dependents[il].insert(ir);

                                    is_match = true;
                                    return;
                                }
                            }); // is_const_r
                            if (is_match)
                                return;

                        }); // view_r
                        if (is_match)
                            return;

                    }); // kernel_r
                    if (is_match)
                        return;

                    // if entry wasn't added yet, map it to null
                    if (is_const_l)
                    { // input
                        inputs.emplace(std::make_tuple(il, jl), std::make_tuple(null_v, null_v));
                    }
                    else
                    { // output
                        outputs.emplace(std::make_tuple(il, jl), std::make_tuple(null_v, null_v));
                    }
                }); // is_const_l, is_match

            }); // view_l

        }); // kernel_l

        printf("\ninputs\n");
        for (const auto& item : inputs)
        {
            const auto& key = item.first;
            std::cout << "(" << std::get<0>(key) << ", " << std::get<1>(key) << ")\n";
        }

        printf("\noutputs\n");
        for (const auto& item : outputs)
        {
            const auto& key = item.first;
            std::cout << "(" << std::get<0>(key) << ", " << std::get<1>(key) << ")\n";
        }

        printf("\ndata dependencies\n");
        for (const auto& item : inputs)
        {
            const auto& key   = item.first;
            const auto& value = item.second;
            if ((std::get<0>(value) != null_v) && (std::get<1>(value) != null_v))
                std::cout << "(" << std::get<0>(value) << ", " << std::get<1>(value) << ")->("
                          << std::get<0>(key) << ", " << std::get<1>(key) << ")\n";
        }

        // build the Directed Acyclic Graph
        iter_tuple(kernels_,
                   [&]<typename KernelType>(size_t i, KernelType& k)
        {
            std::vector<size_t> prev(depends_on[i].begin(), depends_on[i].end());
            std::vector<size_t> next(dependents[i].begin(), dependents[i].end());
            int indegree = prev.size();
            graph.push_back(Node { i, prev, next, indegree });
        });

        // topological search to find all valid kernel_sequences and to
        // ensure there are no circular dependencies
        try
        {
            top_search();
        }
        catch (const std::exception& e)
        {
            std::cerr << "Error: " << e.what() << "\n";
            // exit here
        }

        // identify indpendent streams


        printf("\nkernel dependencies\n");
        for (const auto& node : graph)
        {
            for (size_t i : node.next)
            {
                Node& child = graph[i];
                std::cout << node.i << "->" << child.i << "\n";
            }
        }
        printf("\nkernel sequences\n");
        for (const auto& kseq : kernel_sequences)
        {
            bool first = true;
            for (const auto& i : kseq)
            {
                if (!first)
                    std::cout << " ";
                std::cout << i;
                first = false;
            }
            std::cout << "\n";
        }

        // get device options
        iter_tuple(kernels_, [&]<typename KernelType>(size_t i, KernelType& k) {
            devices.push_back(k.options_.devices);
        });


        // then iterate over the kernels to grab how many additinal IDs to grab with parameters
        iter_tuple(kernels_,
                   [&]<typename KernelType>(size_t i_kernel, KernelType& k_temp)
        {
            // get the number of device execution policies that are stored for this kernel
            kernel_chain_device_vals.push_back(k_temp.n_device_execution_policies_);
        });

        std::cout << "Kernel device chain sizes: ";
        for (auto& ii : kernel_chain_device_vals)
        {
            std::cout << " " << ii;
        }
        std::cout << std::endl;


        // init kernel execution chains
        for (auto& kseq : kernel_sequences)
        {
            for (std::vector<DeviceSelector> dev : cartesian_product(reorder_vector(devices, kseq)))
            {
                std::vector<KernelSelector> kernel_chain;
                for (size_t i = 0; i < kseq.size(); i++)
                {
                    size_t kernel_id             = kseq[i];
                    DeviceSelector kernel_device = dev[i];
                    kernel_chain.push_back({ kernel_id, kernel_device });
                }
                kernel_chains.push_back(kernel_chain);
            }
        }

        // now that chains are created, create a list of IDs based on the number
        kernel_chain_ids.resize(kernel_chains.size());
        std::iota(kernel_chain_ids.begin(), kernel_chain_ids.end(), 0);

        // ==============
        // set up data transfers

        // make sure to update the size of the stored number of transfers for each chain
        chains_total_input_transfers_d2h.resize(kernel_chains.size());
        std::fill(chains_total_input_transfers_d2h.begin(),
                  chains_total_input_transfers_d2h.end(),
                  0);
        chains_total_output_transfers_d2h.resize(kernel_chains.size());
        std::fill(chains_total_output_transfers_d2h.begin(),
                  chains_total_output_transfers_d2h.end(),
                  0);
        chains_total_input_transfers_h2d.resize(kernel_chains.size());
        std::fill(chains_total_input_transfers_h2d.begin(),
                  chains_total_input_transfers_h2d.end(),
                  0);
        chains_total_output_transfers_h2d.resize(kernel_chains.size());
        std::fill(chains_total_output_transfers_h2d.begin(),
                  chains_total_output_transfers_h2d.end(),
                  0);


        // 0: loop over kernel chains via id for additional storage
        for (auto& i_chain : kernel_chain_ids)
        // for (std::vector<KernelSelector>& kernel_chain : kernel_chains)
        { // kernel_chain
            std::vector<KernelSelector>& kernel_chain = kernel_chains[i_chain];

            // 1: loop over kernels in this chain
            for (size_t ksel_l_id = 0; ksel_l_id < kernel_chain.size(); ksel_l_id++)
            { // ksel_l

                KernelSelector& ksel_l = kernel_chain[ksel_l_id];
                size_t _il             = ksel_l.kernel_id;

                std::vector<size_t> outputs_to_add;
                std::vector<size_t> inputs_to_add;

                // 2: find this kernel
                find_tuple(kernels_,
                           _il,
                           [&]<typename KernelTypeL>(KernelTypeL& kernel_l) { // kernel_l
                    // 3: loop over host views in left kernel
                    iter_tuple(std::get<0>(kernel_l.data_views_),
                               [&]<typename ViewTypeL>(size_t jl, ViewTypeL& view_l) { // view_l
                        // for every data param, check if needs to be copied to a different
                        // device default to the same device as the kernel
                        DeviceSelector device = ksel_l.kernel_device;

                        // first, is this view an output?
                        bool is_output = false;
                        for (const auto& item : outputs)
                        {
                            const auto& key = item.first;
                            if ((_il == std::get<0>(key)) && (jl == std::get<1>(key)))
                            {
                                is_output = true;
                                break;
                            }
                        }

                        // NOTE, we could handle pre-moves here for device inputs on the
                        // first kernel

                        // 4: only outputs may need to be copied
                        if (is_output)
                        {

                            // if this is the last kernel we need to copy outputs back to
                            // the host
                            if (ksel_l_id >= kernel_chain.size() - 1)
                            {

                                device = DeviceSelector::HOST;

                                // 5: this is not the last kernel so it may have dependents
                            }
                            else
                            {

                                // look for downstream dependents
                                std::vector<size_t> dependents = std::vector<size_t>();
                                for (const auto& item : inputs)
                                {
                                    const auto& key   = item.first;
                                    const auto& value = item.second;

                                    // find which kernels depend on this left view
                                    if ((_il == std::get<0>(value)) && (jl == std::get<1>(value)))
                                    {

                                        // we only care about the kernel now
                                        size_t ir = std::get<0>(key);
                                        dependents.push_back(ir);
                                    }
                                }

                                // if no dependents, the data can be safely moved back to
                                // the host
                                if (dependents.size() == 0)
                                {
                                    device = DeviceSelector::HOST;

                                    // 6: at least 1 dependent, check if device is different
                                }
                                else
                                {
                                    // 7: loop over the rest of the kernel chain and find
                                    // dependents in order
                                    for (size_t ksel_r_id = ksel_l_id + 1;
                                         ksel_r_id < kernel_chain.size();
                                         ksel_r_id++)
                                    {
                                        KernelSelector& ksel_r = kernel_chain[ksel_r_id];
                                        size_t ir              = ksel_r.kernel_id;
                                        bool is_dependent      = false;

                                        // check if this right kernel is in the list of
                                        // dependents
                                        for (size_t _ir : dependents)
                                        {
                                            if (_ir == ir)
                                            {
                                                is_dependent = true;
                                                break;
                                            }
                                        }
                                        // if this kernel isn't a dependent, skip it!
                                        if (!is_dependent)
                                            continue;

                                        // check if ANY downstream dependents have a
                                        // different device than this kernel
                                        if (ksel_r.kernel_device != device)
                                        {

                                            // OK, we ACTUALLY DO need to move this data
                                            device = ksel_r.kernel_device;

                                            // TODO: also need to store which "local" Id
                                            // matches
                                        }

                                    } // 7

                                } // 6

                            } // 5

                        } // 4

                        // now we need to find the index of each of the kernel views in the
                        // original views tuple 4: loop over the views in the views tuple
                        size_t j = 0;
                        iter_tuple(views_,
                                   [&]<typename ViewType>(size_t _j, ViewType& _views)
                        {
                            // note "_views" is actually a tuple of 3 views, one for each allocation
                            // type

                            // compare host views
                            auto view = std::get<0>(_views);
                            if (view_l.data() == view.data())
                                j = _j;
                        }); // 4

                        if (is_output)
                        {
                            // now we know the device this view will need to be copied to
                            ksel_l.output_device.push_back(device);

                            // store the index
                            ksel_l.output_id_local.push_back(jl);
                            ksel_l.output_id.push_back(j);

                            outputs_to_add.push_back(j);

                            // update the number of output transfers
                            if (device == DeviceSelector::HOST)
                                chains_total_output_transfers_d2h[i_chain] += 1;
                            else
                                chains_total_output_transfers_h2d[i_chain] += 1;
                        }
                        else
                        {
                            // store the index
                            ksel_l.input_id_local.push_back(jl);
                            ksel_l.input_id.push_back(j);

                            // update the number of input transfers
                            if (device == DeviceSelector::HOST)
                                chains_total_input_transfers_d2h[i_chain] += 1;
                            else
                                chains_total_input_transfers_h2d[i_chain] += 1;

                            inputs_to_add.push_back(j);
                        }

                    }); // 3
                });     // 2

                // only work with data dependencies if it's the first chain
                if (i_chain == 0)
                {
                    // now we add the outputs and inputs to the data graph
                    // NOTE: Assume that all inputs handle all outputs, as the kernel requires them
                    for (auto& inp : inputs_to_add)
                    {
                        for (auto& outp : outputs_to_add)
                        {
                            data_graph.add_edge(inp, outp, _il);
                        }
                    }
                }

#ifdef DYNTUNE_DEBUG_ENABLED_FORCED
                std::cout << "Chain, Kernel (" << i_chain << ", " << _il
                          << ") Output IDs (global): ";
                for (auto opid : ksel_l.output_id)
                {
                    std::cout << opid << " ";
                }
                std::cout << std::endl;

                std::cout << "Chain, Kernel (" << i_chain << ", " << _il
                          << ") Input IDs (global): ";
                for (auto opid : ksel_l.input_id)
                {
                    std::cout << opid << " ";
                }
                std::cout << std::endl;

                std::cout << "Chain, Kernel (" << i_chain << ", " << _il
                          << ") Output IDs (local): ";
                for (auto opid : ksel_l.output_id_local)
                {
                    std::cout << opid << " ";
                }
                std::cout << std::endl;

                std::cout << "Chain, Kernel (" << i_chain << ", " << _il << ") Input IDs (local): ";
                for (auto opid : ksel_l.input_id_local)
                {
                    std::cout << opid << " ";
                }
                std::cout << std::endl;
#endif

            } // 1

        } // 0

        // with the vector of kernels now created, we can create a list of IDs to use to store more
        // information about the profiling

        std::cout << std::endl << "kernel chains" << std::endl;
        // for (std::vector<KernelSelector> kernel_chain : kernel_chains) {
        for (uint32_t i_chain : kernel_chain_ids)
        {

            // NOTE: this copies, should it grab by reference?
            std::vector<KernelSelector> kernel_chain = kernel_chains[i_chain];

            bool first_k = true;
            for (KernelSelector ksel : kernel_chain)
            {
                bool first_dp = true;
                if (first_k)
                {
                    std::cout << "Chain " << std::setw(4) << i_chain << ": ";
                    first_k = false;
                }
                else
                    std::cout << " ";
                std::cout << "(" << ksel.kernel_id << ", " << ksel.kernel_device << ", (";
                for (DeviceSelector output_device : ksel.output_device)
                {
                    if (first_dp)
                        first_dp = false;
                    else
                        std::cout << ",";
                    std::cout << output_device;
                }
                std::cout << "))";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Kernel data dependency graph: " << std::endl;
        data_graph.print_graph_normal();
        std::cout << std::endl;

    } // end build_data_graph

    void get_timing_index() {}

    void build_expanded_chains()
    {
        // we need to calculate how many total options we're dealing with, and that includes
        // knowing how many options there are, so we need to iterate over the chains themselves and
        // then verify some things

        uint32_t total_chains = 0;

        std::vector<std::vector<KernelWithOptionsSelector>> true_selection;

        for (uint32_t i_chain : kernel_chain_ids)
        { // chain loop
            std::vector<KernelSelector>& kernel_chain = kernel_chains[i_chain];

            std::vector<std::vector<size_t>> temp;

            for (KernelSelector ksel : kernel_chain)
            { // loop through the kernels themselves
                size_t k_id                  = ksel.kernel_id;
                DeviceSelector kernel_device = ksel.kernel_device;

                // then based on the kernel ID, get the number of options from
                // kernel_chain_device_vals
                size_t n_k_options = kernel_chain_device_vals[k_id];

                std::vector<size_t> other_to_fill;

                if (kernel_device == DeviceSelector::DEVICE)
                {
                    // then check for the total number of values
                    if (n_k_options > 1)
                        for (uint32_t k_option = 0; k_option < n_k_options; k_option++)
                            other_to_fill.push_back(k_option);
                    else
                        other_to_fill.push_back(0);
                }
                else
                    other_to_fill.push_back(0);

                // now we can put this on the list
                temp.push_back(other_to_fill);
            }

            // create all possible permutations of parameters
            using PermStorageType = std::vector<std::vector<std::size_t>>;
            PermStorageType permutation_storage;

            for (const auto& ele : temp[0])
            {
                permutation_storage.push_back({ ele });
            }

            for (std::size_t i_inner = 1; i_inner < temp.size(); i_inner++)
            {
                PermStorageType new_perm_storage;

                for (const auto& comb : permutation_storage)
                {
                    for (const auto& ele : temp[i_inner])
                    {
                        std::vector<std::size_t> newComb = comb;
                        newComb.push_back(ele);
                        new_perm_storage.push_back(newComb);
                    }
                }

                // make sure to update the storage by swapping!
                permutation_storage = new_perm_storage;
            }
            kernel_chain_device_permutations_bounds.push_back(total_chains);
            total_chains += permutation_storage.size();

            // now we have the permutation storage all handled and taken into account
            // just gotta store it all in the kernels so it's accessible
            kernel_chain_device_permutations.push_back(permutation_storage);
        }

        total_num_permutations = total_chains;

#if 0
        std::cout << "ALL PERMUTATIONS: " << std::endl;
        for (const auto& aa : kernel_chain_device_permutations)
        {
            std::cout << "New Chain: ";
            for (const auto& bb : aa)
            {
                for (const auto& cc : bb)
                {
                    std::cout << cc << " ";
                }
                std::cout << ", ";
            }
            std::cout << std::endl;
        }
        std::cout << "Calculated number of total permutations across all chains: "
                  << total_num_permutations << std::endl;
#endif
        std::cout << "Notice: We've discovered **" << total_num_permutations
                  << "** total permutations1" << std::endl;
    }

  public:
    void operator()()
    {
        // std::cout << "Running algorithm attempt id = " << total_operations_run << std::endl;

#ifdef DYNTUNE_ENABLE_ORDER_SHUFFLE
        // std::cout << "Shuffling kernel chains..." << std::endl;
        std::random_shuffle(kernel_chain_ids.begin(), kernel_chain_ids.end());

        // we can also shuffle all of the different permutations as well, maybe want an ID for both?
        // TODO:
#endif

#if DEBUG_ENABLED
        std::cout << "Executing chains in the following order: " << std::endl << "    ";
        for (uint32_t i_chain : kernel_chain_ids)
        {
            std::cout << i_chain << " ";
        }
        std::cout << std::endl << std::endl << std::endl;
#endif

        // set up the execution time vector, if the size isn't already correct
        // if (chain_times.size() != kernel_chains.size())
        //     chain_times.resize(kernel_chains.size(), 0.0);
        if (chain_times.size() != total_num_permutations)
            chain_times.resize(total_num_permutations, 0.0);

        // same with full elapsed times
        // if (chain_elapsed_times.size() != kernel_chains.size())
        //     chain_elapsed_times.resize(kernel_chains.size(), 0.0);
        if (chain_elapsed_times.size() != total_num_permutations)
            chain_elapsed_times.resize(total_num_permutations, 0.0);

        // 0: loop over kernel chains
        // for (std::vector<KernelSelector> kernel_chain : kernel_chains)
        for (uint32_t i_chain : kernel_chain_ids)
        { // kernel_chain

#ifdef DYNTUNE_SINGLE_CHAIN_RUN
            if (i_chain != selected_only_run)
            {
                std::cout << "Skipping chain number: " << i_chain << std::endl;
                continue;
            }
#endif
            std::vector<KernelSelector> kernel_chain = kernel_chains[i_chain];

            // then iterate over the number of permutations
            for (int i_perm = 0; i_perm < kernel_chain_device_permutations[i_chain].size();
                 i_perm++)
            {
                auto perm = kernel_chain_device_permutations[i_chain][i_perm];
                // then figure out which main ID we're at to know where to store the timer
                // information
                std::size_t perm_main_id =
                  kernel_chain_device_permutations_bounds[i_chain] + i_perm;

                for (int i_run = 0; i_run < chain_runs; i_run++)
                {
                    // init timer
                    double elapsed = 0.0;
                    Kokkos::Timer timer;
                    Kokkos::Timer timer_all;
                    timer_all.reset(); // start the timer

                    // first selector may need to copy inputs
                    bool first = true;

                    // trackers for input and output transfers of data
                    uint32_t tracked_input_transfers  = 0;
                    uint32_t tracked_output_transfers = 0;

                    // 1: iterate through the chain
                    for (KernelSelector ksel : kernel_chain)
                    { // ksel, i

                        size_t i                     = ksel.kernel_id;
                        DeviceSelector kernel_device = ksel.kernel_device;

                        // 2: find this kernel
                        find_tuple(kernels_,
                                   i,
                                   [&]<typename KernelType>(KernelType& k) { // k
                            // get the data views
                            auto views_h = std::get<0>(k.data_views_);
                            auto views_d = std::get<1>(k.data_views_);

                            do_input_data_transfer(ksel,
                                                   k,
                                                   i,
                                                   kernel_device,
                                                   elapsed,
                                                   timer,
                                                   tracked_input_transfers,
                                                   i_chain);

                            // execute the kernel
                            timer.reset(); // start the timer
                            // then don't forget to run the kernel at the perm ID
                            k(kernel_device, perm[i]);
                            elapsed += timer.seconds();

                            // copy outputs
                            do_output_data_transfer(ksel,
                                                    k,
                                                    i,
                                                    kernel_device,
                                                    elapsed,
                                                    timer,
                                                    tracked_input_transfers,
                                                    i_chain);


                        }); // 2

                    } // 1

                    // store the execution times in the vectors
                    double chain_time = timer_all.seconds();
                    // chain_times[i_chain] += chain_time;
                    // chain_elapsed_times[i_chain] += elapsed;
                    chain_times[perm_main_id] += chain_time;
                    chain_elapsed_times[perm_main_id] += elapsed;


                    // execute the validation function (could be null)
#ifdef DYTUNE_DEBUG_ENABLED
                    std::cout << "Finished chain " << i_chain << std::endl;
                    std::cout << "Expected transfers (d2h), input: "
                              << chains_total_input_transfers_d2h[i_chain]
                              << " output: " << chains_total_output_transfers_d2h[i_chain]
                              << std::endl;
                    std::cout << "Expected transfers (h2d), input: "
                              << chains_total_input_transfers_h2d[i_chain]
                              << " output: " << chains_total_output_transfers_h2d[i_chain]
                              << std::endl;
                    std::cout << "Tracked transfers, input: " << tracked_input_transfers
                              << " output: " << tracked_output_transfers << std::endl;
#endif

                    { // debug print
                        /*
                        bool success = true;
                        for (KernelSelector ksel : kernel_chain) {
                            for (size_t j : ksel.output_id) {
        //if (j > 4) continue;
                                iter_tuple(views_, [&]<typename ViewType>(size_t _j, ViewType&
        _views) { if (_j == j)
                                {
                                    auto view = std::get<0>(_views);
                                    //if (view.rank() == 1) {
                                        auto N = view.extent(0);
                                        //printf("\n");
                                        for (auto i = 0; i < N; i++) {
                                            //printf("[%d,0,%d] %f\n", j, i, view(i));
                                            if (view(i) == 0) success = false;
                                        }
                                    //}
                                }});
                            }
                        }
                        printf("RESULT: time=%f, success=%s\n", chain_time, (success) ? "true" :
        "false");
                        */
                        // printf("RESULT: ops=%f, all=%f\n", elapsed, chain_time);
                    }

                    // run a stored validation function provided by the user
                    validation_function();
                }
            }

        } // 0

        total_operations_run++;

    } // end operator()

    void print_results(bool sort_results    = true,
                       std::size_t truncate = std::numeric_limits<std::size_t>::max(),
                       std::ostream& outs   = std::cout)
    {
        // this just goes through the kernels and prints the timing results
        outs << "==========================" << std::endl;
        outs << "===== Timing results =====" << std::endl << std::endl;

        outs << "Total number of chains run: " << kernel_chain_ids.size() << std::endl;
        outs << "Total number of permutations: " << total_num_permutations << std::endl;
        outs << "Number of times run: " << total_operations_run << std::endl;
        outs << "Number of times each chain run: " << chain_runs << std::endl;
        outs << "Total number of times run: " << total_operations_run * chain_runs << std::endl;


        outs << "Profiling results (avg): (chain_id, ops_time, total_time):" << std::endl;

        if (sort_results)
        {
            // std::vector<size_t> sorted_ids(kernel_chains.size());
            // std::iota(sorted_ids.begin(), sorted_ids.end(), 0);

            std::vector<size_t> sorted_ids(total_num_permutations);
            std::iota(sorted_ids.begin(), sorted_ids.end(), 0);

            std::stable_sort(sorted_ids.begin(), sorted_ids.end(), [this](size_t i1, size_t i2) {
                return this->chain_times[i1] < this->chain_times[i2];
            });

            std::size_t num_to_print = sorted_ids.size();
            if (num_to_print > 25)
            {
                num_to_print = 25;
            }

            for (std::size_t ii = 0; ii < num_to_print; ii++)
            {
                std::size_t i_chain = sorted_ids[ii];
                double chain_time   = chain_times[i_chain] / total_operations_run / chain_runs;
                double total_time =
                  chain_elapsed_times[i_chain] / total_operations_run / chain_runs;
                outs << "Chain " << std::setw(4) << i_chain << "\t" << std::scientific << chain_time
                     << "\t" << total_time << std::endl;
            }
            // for (auto i_chain : sorted_ids)
            // {
            //     double chain_time = chain_times[i_chain] / total_operations_run / chain_runs;
            //     double total_time =
            //       chain_elapsed_times[i_chain] / total_operations_run / chain_runs;
            //     outs << "Chain " << std::setw(4) << i_chain << "\t" << std::scientific
            //               << chain_time << "\t" << total_time << std::endl;
            // }
        }
        else
        {
            for (size_t i_chain = 0; i_chain < kernel_chains.size(); i_chain++)
            {
                double chain_time = chain_times[i_chain] / total_operations_run / chain_runs;
                double total_time =
                  chain_elapsed_times[i_chain] / total_operations_run / chain_runs;
                outs << "Chain " << std::setw(4) << i_chain << "\t" << std::scientific << chain_time
                     << "\t" << total_time << std::endl;
            }
        }

        outs << std::endl << "==========================" << std::endl;
    }

    template<typename KernelType>
    void do_input_data_transfer(KernelSelector& kernel_selector,
                                KernelType& kernel,
                                size_t& kernel_id,
                                DeviceSelector& kernel_device,
                                double& elapsed,
                                Kokkos::Timer& timer,
                                uint32_t& tracked_input_transfers,
                                uint32_t& chain_id)
    {

        // get the kernel views
        auto views_h = std::get<0>(kernel.data_views_);
        auto views_d = std::get<1>(kernel.data_views_);

        // copy the data over only if the device is active
        if (kernel_device != DeviceSelector::DEVICE)
        {
            return;
        }

        // otherwise we're good to continue
        for (size_t j_it = 0; j_it < kernel_selector.input_id_local.size(); j_it++)
        {
            size_t j_local  = kernel_selector.input_id_local[j_it];
            size_t j_global = kernel_selector.input_id[j_it];


            // make sure we skip this **if** its a dependent, as the data is always moved
            // where it needs to be after execution
            // TODO: this needs to be handled later depending on how we decide to move data around

            DataDependencyGraph::Node* j_dep_node = data_graph.find_node(j_global);

            if (j_dep_node == nullptr)
            {
                throw std::runtime_error("Global input node couldn't be found: kernel_global=" +
                                         std::to_string(j_global));
            }

            bool skip_transfer = false;

            // iterate through previous nodes to check for any dependencies based on kernel id
            for (auto prev_pair : j_dep_node->prev)
            {
                // if any of the incoming edges are *not* this kernel, then it means this is an
                // output elsewhere and will be moved at the end of the kernel execution
                if (prev_pair.first != kernel_id)
                {
                    skip_transfer = true;
                    break;
                }
            }

            if (skip_transfer)
            {

#ifdef DYNTUNE_DEBUG_ENABLED_OFF
                std::cout << "chain_" << chain_id << ": Kernel " << kernel_id
                          << " reporting that it's *NOT* copying data: " << j_local << " (global "
                          << j_global << ")" << std::endl;
#endif
                continue;
            }

            // do the data transfer
            transfer_data_host_to_device(j_it, kernel.data_views_, elapsed, timer);

        } // end input list iteration
    }

    template<typename KernelType>
    void do_output_data_transfer(KernelSelector& kernel_selector,
                                 KernelType& kernel,
                                 size_t& kernel_id,
                                 DeviceSelector& kernel_device,
                                 double& elapsed,
                                 Kokkos::Timer& timer,
                                 uint32_t& tracked_input_transfers,
                                 uint32_t& chain_id)
    {
        // get the kernel views
        auto views_h = std::get<0>(kernel.data_views_);
        auto views_d = std::get<1>(kernel.data_views_);


        for (size_t j_it = 0; j_it < kernel_selector.output_id_local.size(); j_it++)
        {
            size_t j_local             = kernel_selector.output_id_local[j_it];
            size_t j_global            = kernel_selector.output_id[j_it];
            DeviceSelector view_device = kernel_selector.output_device[j_it];

            // no need to copy if the data is already on the correct device
            if (view_device == kernel_device)
                continue;

            if (view_device == DeviceSelector::DEVICE)
            {
                // if the device is supposed to be the device, then we want to perform the copy to
                // device
                transfer_data_host_to_device(j_local, kernel.data_views_, elapsed, timer);
            }
            else
            {
                // otherwise, we want todo our copy from device to host
                transfer_data_device_to_host(j_local, kernel.data_views_, elapsed, timer);
            }
        }

        //
    }

}; // end Algorithm
