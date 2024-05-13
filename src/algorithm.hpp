#pragma once

#include "common.hpp"
#include "kernel.hpp"

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
    constexpr Algorithm(std::tuple<KernelTypes&...> kernels)
      : kernels_(kernels)
    {
        build_data_graph();

#ifdef NDEBUG
        iter_tuple(kernels_,
                   []<typename KernelType>(size_t i, KernelType& kernel)
                   { printf("Registered Kernel: %s\n", kernel.kernel_name_.c_str()); });
#endif
    };
    ~Algorithm() {};

    // the core of this class is a tuple of kernels
    std::tuple<KernelTypes&...> kernels_;

    using index_pair = std::tuple<size_t, size_t>;
    using index_map  = std::map<index_pair, index_pair>;
    index_map inputs;
    index_map outputs;

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

    // build the DataGraph from chain of Kernels
    // template<typename TupleType>
    // auto build_data_graph (TupleType& kernels)
    //template<typename... KernelTypes>
    //void build_data_graph(std::tuple<KernelTypes&...> kernels)
    void build_data_graph()
    {
        //using index_pair = std::tuple<size_t, size_t>;
        //using index_map  = std::map<index_pair, index_pair>;
        size_t null_v    = std::numeric_limits<std::size_t>::max();

        // create an empty inputs and outputs for each
        //index_map inputs;
        //index_map outputs;

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

    //#ifdef NDEBUG
        printf("\ninputs\n");
        for (const auto& item : inputs)
        {
            const auto& key   = item.first;
            const auto& value = item.second;
            std::cout << "Key: (" << std::get<0>(key) << ", " << std::get<1>(key) << "), Value: ("
                    << std::get<0>(value) << ", " << std::get<1>(value) << ")\n";
        }
        printf("\noutputs\n");
        for (const auto& item : outputs)
        {
            const auto& key   = item.first;
            const auto& value = item.second;
            std::cout << "Key: (" << std::get<0>(key) << ", " << std::get<1>(key) << "), Value: ("
                    << std::get<0>(value) << ", " << std::get<1>(value) << ")\n";
        }
    //#endif

        /*
        // now we have maps of all data param connections!

        // next, loop over kernels and make a node for each kernel
        //   how to determine number of inputs and outputs for each kernel?
        //   should I have counted them?
        //   or should we just use vectors?

        // should outputs will null destinations be automatically copied back to the host?
        */
    } // end build_data_graph

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