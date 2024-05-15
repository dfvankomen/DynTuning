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

/*
class Graph
{
public:
    Graph(const std::vector<Node>& nodes) : nodes_(nodes)
    {
        buildAdjacencyMap();
    }

    std::vector<std::vector<Node*>> generateSequences()
    {
        std::vector<std::vector<Node*>> result;
        std::vector<Node*> currentSequence;
        std::unordered_map<Node*, int> inDegree;

        // Initialize in-degree for each node
        for (const Node& node : nodes_)
        {
            inDegree[const_cast<Node*>(&node)] = node.prev.size();
        }

        // Initialize queue with nodes having in-degree 0
        std::queue<Node*> q;
        for (const Node& node : nodes_)
        {
            if (inDegree[const_cast<Node*>(&node)] == 0)
                q.push(const_cast<Node*>(&node));
        }

        // Perform topological sort
        while (!q.empty())
        {
            Node* current = q.front();
            q.pop();
            currentSequence.push_back(current);

            for (Node* child : adjacencyMap_[current])
            {
                if (--inDegree[child] == 0)
                    q.push(child);
            }
        }

        // Generate all possible sequences
        generateAllSequences(currentSequence, result, 0);

        return result;
    }

private:
    void buildAdjacencyMap()
    {
        for (const Node& node : nodes_)
        {
            for (Node* child : node.next)
            {
                adjacencyMap_[const_cast<Node*>(&node)].push_back(child);
            }
        }
    }

    void generateAllSequences(std::vector<Node*>& currentSequence,
                              std::vector<std::vector<Node*>>& result,
                              size_t index)
    {
        if (index == nodes_.size())
        {
            result.push_back(currentSequence);
            return;
        }

        for (Node* nextNode : adjacencyMap_[currentSequence[index]])
        {
            currentSequence.push_back(nextNode);
            generateAllSequences(currentSequence, result, index + 1);
            currentSequence.pop_back();
        }
    }

    const std::vector<Node>& nodes_;
    std::unordered_map<Node*, std::vector<Node*>> adjacencyMap_;
};

*/

// main algorithm object
template<typename... KernelTypes>
class Algorithm
{
  public:
    // constructor should initialize and empty vector
    constexpr Algorithm(std::tuple<KernelTypes&...> kernels)
      : kernels_(kernels)
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

  public:
    // data dependencies define kernel dependencies
    std::vector<Node> graph;
    std::vector<std::vector<size_t>> sequences;

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

                // store result
                if (seq.size() == graph.size())
                    sequences.push_back(seq);

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

    } // end build_data_graph

    void operator() ()
    {
        for (auto& seq : sequences) {
            for (auto il : seq) {
                auto kl = std::get<il>(kernels_);
                for (auto ir : seq) {
                    if (ir <= il) 
                        continue;
                    auto kr = std::get<ir>(kernels_);
                    // START HERE
                    // among the data dependencies of left kernel
                    // find the next right kernel
                    // will get views from that

                } // end loop over right kernels
            } // end loop over left kernels
        } // end loop over sequences
    }

    //START HERE
    // next data dependency now depends on which sequence you use
    // for each kernel0
    //   for each param0 in the kernel0
    //     determine next view that needs this data
    //     if param0 is in an output of kernel0
    //       if index inputs map is valid (less than (18446744073709551615, 18446744073709551615))
    //         get the kernel1 struct from this same array and check its device
    //           if device is device // then deepcopy
    //             get the device and tmp space view references from kernel1
    //             if tmp views are valid
    //               2-step deepcopy
    //             else
    //               1-step deepcopy
    //       else
    //         check the device of this kernel from the current struct
    //         if device is device // then deepcopy
    //             get the device and tmp space view references from kernel0
    //             if tmp views are valid
    //               2-step deepcopy to host view
    //             else
    //               1-step deepcopy to host view

    // struct of each kernel needs:
    //   // kernel id is same as struct index
    //   device_selector
    //   thread count

    // for each iteration of the algorithm generate a new set of these kernel structs

}; // end Algorithm

/*
struct ExecutionParams
{
    //DeviceSelector device;
    ExecutionParams(DeviceSelector device)
    : device(device)
    {}
}
*/

/*
// Serves as a "node" in the data/kernel dependency graph
struct DataGraphNode
{
    DataGraphNode(const size_t I)
      : kernel_id(I)
    {
    }

    // The kernel that this node refers to in the chain of kernels
    const size_t I;

    // For each input data parameter for the corresponding kernel, this
    // indicates the upstream kernel index that outputs the data parameter needed by this kernel
    // along with the index of the data parameter in that upstream kernel.
    std::vector<std::tuple<size_t, size_t, size_t>> inputs = {};

    // For each output data parameter for the corresponding kernel, this
    // indicates the downstream kernel indices that need the data parameter as an input along
    // with the index of the data parameter in those downstream kernels.
    std::vector<std::tuple<size_t, size_t, size_t>> outputs = {};

};
*/