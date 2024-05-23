#pragma once

#include "common.hpp"
#include "kernel.hpp"

#include <set>

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
        iter_tuple(kernels_, [&]<typename KernelType>(size_t i, KernelType& k) {
            depends_on.push_back(std::set<size_t>{});
            dependents.push_back(std::set<size_t>{});
        });

        build_graph();

    };
    ~Algorithm() {};

    // the core of this class is a tuple of kernels
    KernelsTuple kernels_;

    // algorithm is reponsible for shuttling data
    ViewsTuple views_;

    // allow reordering
    bool reordering_;

    // available options for devices to run on
    std::vector<std::vector<DeviceSelector>> devices;

    // index maps define data dependencies
    using index_pair = std::tuple<size_t, size_t>;
    using index_map  = std::map<index_pair, index_pair>;
    using adjacency_map = std::vector<std::set<size_t>>;
    index_map     inputs;
    index_map     outputs;
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

  private:

    enum { UNVISITED, INPROGRESS, VISITED };

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
            state = UNVISITED;
        }
    };

    // Function to compute the Cartesian product of arrays
    template<typename T>
    std::vector<std::vector<T>> cartesian_product(const std::vector<std::vector<T>>& arrays) {
        std::vector<size_t> indices(arrays.size(), 0);
        bool changed;
        std::vector<std::vector<T>> result;

        do {
            std::vector<T> tuple;
            for (size_t i = 0; i < arrays.size(); ++i) {
                tuple.push_back(arrays[i][indices[i]]);
            }
            result.push_back(tuple);

            // Update indices
            changed = false;
            for (size_t i = arrays.size(); i-- > 0;) {
                ++indices[i];
                if (indices[i] < arrays[i].size()) {
                    changed = true;
                    break;
                } else {
                    indices[i] = 0;
                }
            }
        } while (changed);

        return result;
    }

    template<typename T, typename IndexType>
    std::vector<std::vector<T>> reorder_vector(
        std::vector<std::vector<T>>& B,
        std::vector<IndexType>& A) {
        std::vector<std::vector<T>> result(B.size());
        for (IndexType i = 0; i < A.size(); ++i)
            result[i] = B[A[i]];
        return result;
    }

    struct KernelSelector
    {
        //kernel
        size_t         kernel_id;     // index of the kernel in the original tuple
        DeviceSelector kernel_device; // device this kernel will run on

        //data
        std::vector<size_t>         input_id      = std::vector<size_t>();
        std::vector<size_t>         output_id     = std::vector<size_t>();
        std::vector<DeviceSelector> output_device = std::vector<DeviceSelector>();
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
    std::vector<double> chain_times = std::vector<double>();

  private:
    // topological search to ensure the graph is acyclic
    void top_search() {
        for (auto &node : graph)
            node.reset();
        std::vector<size_t> kseq;
        top_search_impl(kseq);
    }
    void top_search_impl(std::vector<size_t>& kseq)
    {
        for (auto &node : graph) {
            if ((node.indegree == 0) && (node.state == UNVISITED)) {
                node.state = INPROGRESS;
                for (size_t i : node.next) {
                    Node &child = graph[i];

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
                if (kseq.size() == graph.size()) {
                    kernel_sequences.push_back(kseq);

                    // if no reordering, then just need first sequence
                    if (!reordering_)
                        return;
                }

                // backtrack
                kseq.erase(kseq.end()-1);
                node.state = UNVISITED;
                for (size_t i : node.next) {
                    Node &child = graph[i];
                    child.indegree++;
                }

            } // end node
        } // end loop over nodes
    }
    
    //NOTE
    // the iter_tuple construct is a workaround to access tuple elements with a runtime index
    // the variant idiom has similar limitations, see
    //https://stackoverflow.com/questions/52088928/trying-to-return-the-value-from-stdvariant-using-stdvisit-and-a-lambda-expre

    // build the DataGraph from chain of Kernels
    void build_graph()
    {
        size_t null_v = std::numeric_limits<std::size_t>::max();

        // 0: loop over left kernel
        iter_tuple(kernels_, [&]<typename KernelTypeL>(size_t il, KernelTypeL& kernel_l)
        { // kernel_l

            // 1: loop over left data views
            iter_tuple(std::get<0>(kernel_l.data_views_), [&]<typename ViewTypeL>(size_t jl, ViewTypeL& view_l)
            { // view_l

                // 2: check is_const for jlth data view
                iter_tuple(kernel_l.is_const_, [&](size_t _jl, bool is_const_l) { if (_jl == jl)
                { // is_const_l
                    //bool is_const_l = false;

                    bool is_match = false;

                    // 3: loop over right kernels
                    iter_tuple(kernels_, [&]<typename KernelTypeJ>(size_t ir, KernelTypeJ& kernel_r)
                    { // kernel_r

                        // only look at kernels to the right
                        if (ir <= il) return;

                        // 4: loop over right data views
                        iter_tuple(std::get<0>(kernel_r.data_views_), [&]<typename ViewTypeR>(size_t jr, ViewTypeR& view_r)
                        { // view_r

                            // 5: check is_const for jrth data view
                            iter_tuple(kernel_r.is_const_, [&](size_t _jr, bool is_const_r) { if (_jr == jr)
                            { // is_const_r
                                //bool is_const_r = false;

                                // check if there is a data dependency between these views
                                if ((view_l.data() == view_r.data()) && (!is_const_l) && (is_const_r))
                                {
                                    // view_r depends on view_l
                                    outputs.emplace(std::make_tuple(il, jl), std::make_tuple(ir, jr));
                                    inputs.emplace(std::make_tuple(ir, jr), std::make_tuple(il, jl));

                                    //kernel_r depends on kernel_l
                                    depends_on[ir].insert(il);
                                    dependents[il].insert(ir);

                                    is_match = true;
                                    return;
                                }
                                
                            }}); // is_const_r
                            if (is_match) return;

                        }); // view_r
                        if (is_match) return;

                    }); // kernel_r
                    if (is_match) return;

                    // if entry wasn't added yet, map it to null
                    if (is_const_l) { // input
                        inputs.emplace(std::make_tuple(il, jl), std::make_tuple(null_v, null_v));
                    } else { // output
                        outputs.emplace(std::make_tuple(il, jl), std::make_tuple(null_v, null_v));
                    }

                }}); // is_const_l, is_match

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
                std::cout
                  << "("
                  << std::get<0>(value)
                  << ", "
                  << std::get<1>(value)
                  << ")->("
                  << std::get<0>(key)
                  << ", "
                  << std::get<1>(key)
                  << ")\n";
        }

        // build the Directed Acyclic Graph
        iter_tuple(kernels_, [&]<typename KernelType>(size_t i, KernelType& k) {
            std::vector<size_t> prev(depends_on[i].begin(), depends_on[i].end());
            std::vector<size_t> next(dependents[i].begin(), dependents[i].end());
            int indegree = prev.size();
            graph.push_back(Node{i, prev, next, indegree});
        });

        // topological search to find all valid kernel_sequences and to 
        // ensure there are no circular dependencies
        try {
            top_search();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            //exit here
        }

        // identify indpendent streams


        printf("\nkernel dependencies\n");
        for (const auto& node : graph)
        {
            for (size_t i : node.next) {
                Node &child = graph[i];
                std::cout << node.i << "->" << child.i << "\n";
            }
        }
        printf("\nkernel sequences\n");
        for (const auto& kseq : kernel_sequences)
        {
            bool first = true;
            for (const auto& i : kseq) {
                if (!first)
                    std::cout << " ";
                std::cout << i ;
                first = false;
            }
            std::cout << "\n";
        }

        // get device options
        iter_tuple(kernels_, [&]<typename KernelType>(size_t i, KernelType& k)
        {
            devices.push_back(k.options_.devices);
        });

        // init kernel execution chains
        for (auto& kseq : kernel_sequences)
        {
            for (std::vector<DeviceSelector> dev : cartesian_product(reorder_vector(devices, kseq)))
            {
                std::vector<KernelSelector> kernel_chain;
                for (size_t i = 0; i < kseq.size(); i++)
                {
                    size_t kernel_id = kseq[i];
                    DeviceSelector kernel_device = dev[i];               
                    kernel_chain.push_back({ kernel_id, kernel_device });
                }
                kernel_chains.push_back(kernel_chain);
            }
        }

        // set up data transfers

        // 0: loop over kernel chains
        for (std::vector<KernelSelector>& kernel_chain : kernel_chains)
        { // kernel_chain

            // 1: loop over kernels in this chain
            for (size_t ksel_l_id = 0; ksel_l_id < kernel_chain.size(); ksel_l_id++)
            { // ksel_l

                KernelSelector& ksel_l = kernel_chain[ksel_l_id];
                size_t _il = ksel_l.kernel_id;

                // 2: find this kernel
                iter_tuple(kernels_, [&]<typename KernelTypeL>(size_t il, KernelTypeL& kernel_l) { if (il == _il)
                { // kernel_l
                        
                    // 3: loop over host views in left kernel
                    iter_tuple(std::get<0>(kernel_l.data_views_), [&]<typename ViewTypeL>(size_t jl, ViewTypeL& view_l)
                    { // view_l

                        // for every data param, check if needs to be copied to a different device
                        // default to the same device as the kernel
                        DeviceSelector device = ksel_l.kernel_device;

                        // first, is this view an output?
                        bool is_output = false;
                        for (const auto& item : outputs) {
                            const auto& key = item.first;
                            if ((il == std::get<0>(key)) && (jl == std::get<1>(key))) {
                                is_output = true;
                                break;
                            }
                        }
                        
                        // NOTE, we could handle pre-moves here for device inputs on the first kernel

                        // 4: only outputs may need to be copied
                        if (is_output)
                        { 

                            // if this is the last kernel we need to copy outputs back to the host
                            if (ksel_l_id >= kernel_chain.size()-1) {

                                device = DeviceSelector::HOST;

                            // 5: this is not the last kernel so it may have dependents
                            } else {

                                // look for downstream dependents
                                std::vector<size_t> dependents = std::vector<size_t>();
                                for (const auto& item : inputs) {
                                    const auto& key   = item.first;
                                    const auto& value = item.second;

                                    // find which kernels depend on this left view
                                    if ((il == std::get<0>(value)) && (jl == std::get<1>(value))) {

                                        // we only care about the kernel now
                                        size_t ir = std::get<0>(key);
                                        dependents.push_back(ir);
                                    }
                                }

                                // if no dependents, the data can be safely moved back to the host
                                if (dependents.size() == 0) {
                                    device = DeviceSelector::HOST;
                                
                                // 6: at least 1 dependent, check if device is different
                                } else {

                                    // 7: loop over the rest of the kernel chain and find dependents in order
                                    for (size_t ksel_r_id = ksel_l_id+1; ksel_r_id < kernel_chain.size(); ksel_r_id++)
                                    {
                                        KernelSelector& ksel_r = kernel_chain[ksel_r_id];
                                        size_t ir = ksel_r.kernel_id;
                                        bool is_dependent = false;

                                        // check if this right kernel is in the list of dependents
                                        for (size_t _ir : dependents) {
                                            if (_ir == ir) {
                                                is_dependent = true;
                                                break;
                                            }
                                        }
                                        // if this kernel isn't a dependent, skip it!
                                        if (!is_dependent) continue;

                                        // check if ANY downstream dependents have a different device than this kernel
                                        if (ksel_r.kernel_device != device) {

                                            // OK, we ACTUALLY DO need to move this data
                                            device = ksel_r.kernel_device;
                                        }

                                    } // 7

                                } // 6
                                
                            } // 5

                        } // 4

                        // now we need to find the index of each of the kernel views in the original views tuple
                        // 4: loop over the views in the views tuple
                        size_t j = 0;
                        iter_tuple(views_, [&]<typename ViewType>(size_t _j, ViewType& _views)
                        { // note "_views" is actually a tuple of 3 views, one for each allocation type

                            // compare host views
                            auto view = std::get<0>(_views);
                            if (view_l.data() == view.data()) j = _j;

                        }); // 4

                        if (is_output) {

                            // now we know the device this view will need to be copied to
                            ksel_l.output_device.push_back(device);

                            // store the index
                            ksel_l.output_id.push_back(j);

                        } else {

                            // store the index
                            ksel_l.input_id.push_back(j);

                        }
                        
                    }); // 3
                
                }}); // 2
            
            } // 1
            
        } // 0

        std::random_shuffle(kernel_chains.begin(), kernel_chains.end());
        std::cout << std::endl << "kernel chains" << std::endl;
        for (std::vector<KernelSelector> kernel_chain : kernel_chains) {
            bool first_k = true;
            for (KernelSelector ksel : kernel_chain) {
                bool first_dp = true;
                if (first_k)
                    first_k = false;
                else
                    std::cout << " ";
                std::cout << "(" << ksel.kernel_id << ", " << ksel.kernel_device << ", (";
                for (DeviceSelector output_device : ksel.output_device) {
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

    } // end build_data_graph


public:
    void operator() ()
    {

        // 0: loop over kernel chains
        for (std::vector<KernelSelector> kernel_chain : kernel_chains)
        { // kernel_chain

            // init timer
            double elapsed = 0.0;
            Kokkos::Timer timer;
            Kokkos::Timer timer_all;
            timer_all.reset(); // start the timer

            // first selector may need to copy inputs    
            bool first = true;

            // 1: iterate through the chain
            for (KernelSelector ksel : kernel_chain)
            { // ksel, i

                size_t i = ksel.kernel_id;
                DeviceSelector kernel_device = ksel.kernel_device;

                // 2: find this kernel
                iter_tuple(kernels_, [&]<typename KernelType>(size_t _i, KernelType& k) { if (_i == i)
                { // k

                    // get the data views
                    auto views_h = std::get<0>(k.data_views_);
                    auto views_d = std::get<1>(k.data_views_);

                    // 3: copy inputs for first kernel if needed
                    if ((first) && (kernel_device == DeviceSelector::DEVICE)) {

                        // 4: loop over inputs only
                        for (size_t j : ksel.input_id)
                        {
                            // 5: get the host view
                            iter_tuple(views_h, [&]<typename HostViewType>(size_t jh, HostViewType& view_h) { if (jh == j)
                            { // view_h

                                // 6: get the device view
                                iter_tuple(views_d, [&]<typename DeviceViewType>(size_t jd, DeviceViewType& view_d) { if (jd == j)
                                { // view_d

                                    // copy the data
                                    timer.reset(); // start the timer
                                    Kokkos::deep_copy(view_d, view_h);
                                    elapsed += timer.seconds();

                                }}); // 6

                            }}); // 5

                        }

                        // mark first as done
                        first = false;

                    } // 3
                    
                    // execute the kernel
                    timer.reset(); // start the timer
                    k(kernel_device);
                    elapsed += timer.seconds();

                    // copy outputs

                    // 3: loop over the outputs
                    //for (size_t j : ksel.output_id)
                    for (auto idx = 0; idx < ksel.output_id.size(); idx++)
                    {
                        size_t j = ksel.output_id[idx];
                        DeviceSelector view_device = ksel.output_device[idx];
                        // no need to copy if data is already on the correct device
                        if (view_device == kernel_device) continue;
                        // 4: get the host view
                        iter_tuple(views_h, [&]<typename HostViewType>(size_t jh, HostViewType& view_h) { if (jh == j)
                        { // view_h
                            
                            // 5: get the device view
                            iter_tuple(views_d, [&]<typename DeviceViewType>(size_t jd, DeviceViewType& view_d) { if (jd == j)
                            { // view_d

                                // copy the data, ensure direction is correct
                                timer.reset(); // start the timer
                                if (view_device == DeviceSelector::DEVICE) {
                                    Kokkos::deep_copy(view_d, view_h);
                                } else {
                                    Kokkos::deep_copy(view_h, view_d);
                                }
                                elapsed += timer.seconds();

                            }}); // 5

                        }}); // 4

                    } // 3

                }}); // 2

            } // 1

            // store the execution time
            double chain_time = timer_all.seconds();
            chain_times.push_back(elapsed);

            { // debug print
                /*
                bool success = true;
                for (KernelSelector ksel : kernel_chain) {
                    for (size_t j : ksel.output_id) {
//if (j > 4) continue;
                        iter_tuple(views_, [&]<typename ViewType>(size_t _j, ViewType& _views) { if (_j == j)
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
                printf("RESULT: time=%f, success=%s\n", chain_time, (success) ? "true" : "false");
                */
                printf("RESULT: ops=%f, all=%f\n", elapsed, chain_time);
            }

        } // 0

    } // end operator()

}; // end Algorithm
