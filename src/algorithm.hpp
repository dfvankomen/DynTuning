#pragma once

#include "common.hpp"
#include "kernel.hpp"

#include <set>

// At the end, the algorithm needs to know the "final" output that needs copied to the host
// Data needs moved if 1) it is a kernel input or 2) algorithm output
// Data view deallocation if 1) it is not a downstream input 2) and not algorithm output
// - perhaps use counter for each view (+1 for algorithm output) to know when to deallocate
// it Need algorithm to construct the counters, for example:
//   k.parameters[1] is not const and hence output
//   k2.parameters[0] is const and hence input
// assert(&std::get<1>(k.parameters) == &std::get<0>(k2.parameters));

// main algorithm object
template<typename... KernelTypes>
class Algorithm
{
  public:
    // constructor should initialize and empty vector
    constexpr Algorithm(std::tuple<KernelTypes&...> kernels, bool reordering = false)
      : kernels_(kernels)
      , reordering_(reordering)
    {
        iter_tuple(kernels_, [&]<typename KernelType>(size_t i, KernelType& k) {
            depends_on.push_back(std::set<size_t>{});
            dependents.push_back(std::set<size_t>{});
        });

        build_graph();

        /*
        #ifdef NDEBUG
        iter_tuple(kernels_,
                   []<typename KernelType>(size_t i, KernelType& kernel)
                   { printf("Registered Kernel: %s\n", kernel.kernel_name_.c_str()); });
        #endif
        */
    };
    ~Algorithm() {};

    // the core of this class is a tuple of kernels
    std::tuple<KernelTypes&...> kernels_;

    // options for devices to run on
    std::vector<std::vector<DeviceSelector>> devices;

    // allow reordering
    bool reordering_;

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

    struct KernelExecutor
    {
        size_t kernel_id;
        DeviceSelector device;
        std::vector<DeviceSelector> to_device   = std::vector<DeviceSelector>();
    };

  public:

    // data dependencies define kernel dependencies
    std::vector<Node> graph;
    std::vector<std::vector<size_t>> sequences;
    std::vector<std::vector<KernelExecutor>> kernel_chains;

  private:
    // topological search to ensure the graph is acyclic
    void top_search() {
        for (auto &node : graph)
            node.reset();
        std::vector<size_t> seq;
        top_search_impl(seq);
    }
    void top_search_impl(std::vector<size_t>& seq)
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
                seq.push_back(node.i);
                node.state = VISITED;

                // recurse
                top_search_impl(seq);
                if ((!reordering_) && (sequences.size() == 1))
                    return;

                // store result
                if (seq.size() == graph.size()) {
                    sequences.push_back(seq);

                    // if no reordering, then just need first sequence
                    if (!reordering_)
                        return;
                }

                // backtrack
                seq.erase(seq.end()-1);
                node.state = UNVISITED;
                for (size_t i : node.next) {
                    Node &child = graph[i];
                    child.indegree++;
                }

            } // end node
        } // end loop over nodes
    }
    
    // build the DataGraph from chain of Kernels
    void build_graph()
    {
        size_t null_v = std::numeric_limits<std::size_t>::max();

        // left kernel
        iter_tuple(
        kernels_,
        [&]<typename KernelTypeL>(size_t il, KernelTypeL& kernel_l)
        {
            // left data param
            iter_tuple(
                kernel_l.data_params_,
                [&]<typename ParamTypeL>(size_t jl, ParamTypeL& param_l)
                {
                    bool is_const_l = std::is_const_v<std::remove_reference_t<decltype(param_l)>>;

                    // right kernel
                    bool is_match = false;
                    iter_tuple(
                    kernels_,
                    [&]<typename KernelTypeJ>(size_t ir, KernelTypeJ& kernel_r)
                    {
                        if (ir <= il)
                            return;

                        // right data param
                        iter_tuple(
                            kernel_r.data_params_,
                            [&]<typename ParamTypeR>(size_t jr, ParamTypeR& param_r)
                            {
                                bool is_const_r =
                                std::is_const_v<std::remove_reference_t<decltype(param_r)>>;
                                // printf("(%d %d %d) (%d %d %d)\n", (int) il, (int) jl, (is_const_l) ?
                                // 1 : 0, (int) ir, (int) jr, (is_const_r) ? 1 : 0);

                                // match
                                if ((&param_l == &param_r) && (!is_const_l) && (is_const_r))
                                {
                                    // printf("param %d in kernel %d depends on param %d in kernel
                                    // %d\n",
                                    //   (int) jr, (int) ir, (int) jl, (int) il);
                                    outputs.emplace(std::make_tuple(il, jl), std::make_tuple(ir, jr));
                                    inputs.emplace(std::make_tuple(ir, jr), std::make_tuple(il, jl));

                                    //kernel ir depends on kernel il
                                    depends_on[ir].insert(il);
                                    dependents[il].insert(ir);

                                    is_match = true;
                                    return;
                                }
                            }); // end jr

                        // found a match for this data param
                        if (is_match)
                            return;
                    }); // end ir

                    // found a match for this data param
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
                }); // end jl
        });       // end il

        printf("\ninputs\n");
        for (const auto& item : inputs)
        {
            const auto& key   = item.first;
            //const auto& value = item.second;
            std::cout << "(" << std::get<0>(key) << ", " << std::get<1>(key) << ")\n";
        }

        printf("\noutputs\n");
        for (const auto& item : outputs)
        {
            const auto& key   = item.first;
            //const auto& value = item.second;
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
        iter_tuple(kernels_, [&]<typename KernelType>(size_t i, KernelType& k) {
            std::vector<size_t> prev(depends_on[i].begin(), depends_on[i].end());
            std::vector<size_t> next(dependents[i].begin(), dependents[i].end());
            int indegree = prev.size();
            graph.push_back(Node{i, prev, next, indegree});
        });

        // topological search to find all valid sequences and to 
        // ensure there are no circular dependencies
        try {
            top_search();
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            //exit here
        }
        printf("\nkernel dependencies\n");
        for (const auto& node : graph)
        {
            for (size_t i : node.next) {
                Node &child = graph[i];
                std::cout << node.i << "->" << child.i << "\n";
            }
        }
        printf("\nsequences\n");
        for (const auto& seq : sequences)
        {
            bool first = true;
            for (const auto& i : seq) {
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

        // build execution chains
        for (auto& seq : sequences) {
            for (std::vector<DeviceSelector> dev : cartesian_product(reorder_vector(devices, seq))) {
                std::vector<KernelExecutor> chain;
                for (size_t i = 0; i < seq.size(); i++) {
                    size_t kernel_id = seq[i];
                    DeviceSelector device = dev[i];

                    // for each data param
                    //    what is the next downstream kernel that depends on it?
                    //    if found
                    //        get the device of that kernel
                    iter_tuple(kernels_, [&]<typename KernelType>(size_t i, KernelType& k)
                    {
                        if (kernel_id == i) {

                            // k is current kernel
                        }
                    });
                    
                    chain.push_back({ kernel_id, device });
                }
                kernel_chains.push_back(chain);
            }
        }

        // set up data transfers
        for (std::vector<KernelExecutor>& kernel_chain : kernel_chains) {

            for (size_t ln_id = 0; ln_id < kernel_chain.size(); ln_id++) {

                KernelExecutor& ln = kernel_chain[ln_id];
                size_t kl_id = ln.kernel_id;

                // find this kernel
                iter_tuple(kernels_, [&]<typename KernelTypeL>(size_t il, KernelTypeL& kl)
                {
                    if (il == kl_id) { // found this kernel

                        // for every data param, check if needs to be moved to the DEVICE
                        iter_tuple(kl.data_params_, [&]<typename ParamTypeL>(size_t jl, ParamTypeL& param_l)
                        { //(il, jl)
                            
                            // to_device will only be DEVICE if (il, jl) is an output and
                            //   there is a downstream dependent whose kernel is on DEVICE
                            DeviceSelector device = DeviceSelector::AUTO;

                            // first, is this an output?
                            bool is_output = false;
                            for (const auto& item : outputs) {
                                const auto& key = item.first;
                                if ((il == std::get<0>(key)) && (jl == std::get<1>(key))) {
                                    is_output = true;
                                    break;
                                }
                            }

                            if (is_output) {
                                // (il, jl) is an output, default to HOST

                                device = DeviceSelector::HOST;

                                // if this is the last kernel, end since we need to copy all outputs
                                //   of the last kernel back to the host
                                if (ln_id < kernel_chain.size()-1) {

                                    // since this is an output, look for downstream dependents
                                    //  if any are found, and if any are on the DEVICE, then we
                                    //  need to go ahead and move the data to DEVICE
                                    for (const auto& item : inputs) {
                                        const auto& key   = item.first;
                                        const auto& value = item.second;
                                        if ((il == std::get<0>(value)) && (jl == std::get<1>(value))) {
        
                                            size_t ir = std::get<0>(key); // only care about the kernel now                                        
                                            
                                            // ir is a downstream dependent
                                            // find kernel ir in the current chain and check its device
                                            for (size_t rn_id = ln_id+1; rn_id < kernel_chain.size(); rn_id++) {
                                                KernelExecutor& rn = kernel_chain[rn_id];
                                                if (rn.kernel_id == ir) { // we've found a downstream dependent
                                                    if (rn.device == DeviceSelector::DEVICE) {
                                                        // OK, we ACTUALLY DO need to move this particular data param to the device
                                                        device = DeviceSelector::DEVICE;
                                                    }
                                                    break; // found ir, so stop looking
                                                }
                                            }

                                            // if device is stil HOST, keep looking
                                            if (device == DeviceSelector::DEVICE)
                                                break; // stop looking
                                        }
                                    }
                                }

                            } else if (!is_output) {
                                // (il, jl) is an input, default to same as kernel
                                device = ln.device;
                            }

                            // push this device to the list
                            ln.to_device.push_back(device);
                        });
                    }
                });
            }
        }

        std::cout << std::endl << "kernel chains" << std::endl;
        for (std::vector<KernelExecutor> kernel_chain : kernel_chains) {
            bool first_k = true;
            for (KernelExecutor k : kernel_chain) {
                bool first_dp = true;
                if (first_k)
                    first_k = false;
                else
                    std::cout << " ";
                std::cout << "(" << k.kernel_id << ", " << k.device << ", (";
                for (DeviceSelector to_device : k.to_device) {
                    if (first_dp)
                        first_dp = false;
                    else
                        std::cout << ",";
                    std::cout << to_device;
                }
                std::cout << "))";
            }
            std::cout << std::endl;
        }

    } // end build_data_graph


public:
    void operator() ()
    {
        //loop over kernel_chains
        //for each chain
        //  start timer
        //  for each kernel
        //    execute
        //    move data
        //      if tmp views are valid
        //        2-step deepcopy
        //      else
        //        1-step deepcopy
        //  end timer
        //  store time
    }

}; // end Algorithm
