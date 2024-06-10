#pragma once

#include "common.hpp"
#include "kernel.hpp"

#include <functional>
#include <iomanip>
#include <set>

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
        iter_tuple(
          kernels_,
          [&]<typename KernelTypeL>(size_t il, KernelTypeL& kernel_l) { // kernel_l
              // 1: loop over left data views
              iter_tuple(
                std::get<0>(kernel_l.data_views_),
                [&]<typename ViewTypeL>(size_t jl, ViewTypeL& view_l) { // view_l
                    // 2: check is_const for jlth data view
                    iter_tuple(
                      kernel_l.is_const_,
                      [&](size_t _jl, bool is_const_l)
                      {
                          if (_jl == jl)
                          { // is_const_l
                              // bool is_const_l = false;

                              bool is_match = false;

                              // 3: loop over right kernels
                              iter_tuple(
                                kernels_,
                                [&]<typename KernelTypeJ>(size_t ir,
                                                          KernelTypeJ& kernel_r) { // kernel_r
                                    // only look at kernels to the right
                                    if (ir <= il)
                                        return;

                                    // 4: loop over right data views
                                    iter_tuple(
                                      std::get<0>(kernel_r.data_views_),
                                      [&]<typename ViewTypeR>(size_t jr,
                                                              ViewTypeR& view_r) { // view_r
                                          // 5: check is_const for jrth data view
                                          iter_tuple(
                                            kernel_r.is_const_,
                                            [&](size_t _jr, bool is_const_r)
                                            {
                                                if (_jr == jr)
                                                { // is_const_r
                                                    // bool is_const_r = false;

                                                    // check if there is a data dependency between
                                                    // these views
                                                    if ((view_l.data() == view_r.data()) &&
                                                        (!is_const_l) && (is_const_r))
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
                                  inputs.emplace(std::make_tuple(il, jl),
                                                 std::make_tuple(null_v, null_v));
                              }
                              else
                              { // output
                                  outputs.emplace(std::make_tuple(il, jl),
                                                  std::make_tuple(null_v, null_v));
                              }
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
        iter_tuple(kernels_,
                   [&]<typename KernelType>(size_t i, KernelType& k)
                   { devices.push_back(k.options_.devices); });

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

                // 2: find this kernel
                iter_tuple(
                  kernels_,
                  [&]<typename KernelTypeL>(size_t il, KernelTypeL& kernel_l)
                  {
                      if (il == _il)
                      { // kernel_l

                          // 3: loop over host views in left kernel
                          iter_tuple(
                            std::get<0>(kernel_l.data_views_),
                            [&]<typename ViewTypeL>(size_t jl, ViewTypeL& view_l) { // view_l
                                // for every data param, check if needs to be copied to a different
                                // device default to the same device as the kernel
                                DeviceSelector device = ksel_l.kernel_device;

                                // first, is this view an output?
                                bool is_output = false;
                                for (const auto& item : outputs)
                                {
                                    const auto& key = item.first;
                                    if ((il == std::get<0>(key)) && (jl == std::get<1>(key)))
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
                                            if ((il == std::get<0>(value)) &&
                                                (jl == std::get<1>(value)))
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
                                iter_tuple(
                                  views_,
                                  [&]<typename ViewType>(
                                    size_t _j,
                                    ViewType& _views) { // note "_views" is actually a tuple of 3
                                                        // views, one for each allocation type
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
                                }

                            }); // 3
                      }
                  }); // 2

#ifdef DYNTUNE_DEBUG_ENABLED
                std::cout << "Chain, Kernel " << i_chain << ", " << _il << " Output IDs (global): ";
                for (auto opid : ksel_l.output_id)
                {
                    std::cout << opid << " ";
                }
                std::cout << std::endl;

                std::cout << "Chain, Kernel " << i_chain << ", " << _il << " Input IDs (global): ";
                for (auto opid : ksel_l.input_id)
                {
                    std::cout << opid << " ";
                }
                std::cout << std::endl;

                std::cout << "Chain, Kernel " << i_chain << ", " << _il << " Output IDs (local): ";
                for (auto opid : ksel_l.output_id_local)
                {
                    std::cout << opid << " ";
                }
                std::cout << std::endl;

                std::cout << "Chain, Kernel " << i_chain << ", " << _il << " Input IDs (local): ";
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

    } // end build_data_graph


  public:
    void operator()()
    {
        // std::cout << "Running algorithm attempt id = " << total_operations_run << std::endl;

#ifdef DYNTUNE_ENABLE_ORDER_SHUFFLE
        // std::cout << "Shuffling kernel chains..." << std::endl;
        std::random_shuffle(kernel_chain_ids.begin(), kernel_chain_ids.end());
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
        if (chain_times.size() != kernel_chains.size())
            chain_times.resize(kernel_chains.size(), 0.0);

        // same with full elapsed times
        if (chain_elapsed_times.size() != kernel_chains.size())
            chain_elapsed_times.resize(kernel_chains.size(), 0.0);

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
                    iter_tuple(
                      kernels_,
                      [&]<typename KernelType>(size_t _i, KernelType& k)
                      {
                          if (_i == i)
                          { // k

                              // get the data views
                              auto views_h = std::get<0>(k.data_views_);
                              auto views_d = std::get<1>(k.data_views_);

                              // 3: copy data over to device if it needs it
                              //    NOTE: thsi used to apply only to the first kernel, but the first
                              //    operator was causing data to get skipped further down chains
                              if ((kernel_device == DeviceSelector::DEVICE))
                              {

                                  // 4: loop over inputs only
                                  for (size_t j : ksel.input_id_local)
                                  {
                                      // 5: get the host view
                                      iter_tuple(
                                        views_h,
                                        [&]<typename HostViewType>(size_t jh, HostViewType& view_h)
                                        {
                                            if (jh == j)
                                            { // view_h

                                                // 6: get the device view
                                                iter_tuple(
                                                  views_d,
                                                  [&]<typename DeviceViewType>(
                                                    size_t jd,
                                                    DeviceViewType& view_d)
                                                  {
                                                      if (jd == j)
                                                      { // view_d

                                                    // copy the data
#ifdef DYNTUNE_DEBUG_ENABLED
                                                          std::cout
                                                            << "chain_" << i_chain << ": ker_" << i
                                                            << ":  INPUT -> host-2-device : view "
                                                            << jh << "->" << jd << std::endl;
#endif
                                                          timer.reset(); // start the timer
                                                          // NOTE: remember that deep copy is
                                                          // (destination, source)
                                                          Kokkos::deep_copy(view_d, view_h);
                                                          elapsed += timer.seconds();

                                                          // tick up our successful input transfers
                                                          tracked_input_transfers++;
                                                      }
                                                  }); // 6
                                            }
                                        }); // 5
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
                              // for (size_t j : ksel.output_id)
                              for (auto idx = 0; idx < ksel.output_id_local.size(); idx++)
                              {
                                  size_t j                   = ksel.output_id_local[idx];
                                  DeviceSelector view_device = ksel.output_device[idx];
                                  // no need to copy if data is already on the correct device
                                  if (view_device == kernel_device)
                                      continue;
                                  // 4: get the host view
                                  iter_tuple(
                                    views_h,
                                    [&]<typename HostViewType>(size_t jh, HostViewType& view_h)
                                    {
                                        if (jh == j)
                                        { // view_h

                                            // 5: get the device view
                                            iter_tuple(
                                              views_d,
                                              [&]<typename DeviceViewType>(size_t jd,
                                                                           DeviceViewType& view_d)
                                              {
                                                  if (jd == j)
                                                  { // view_d

                                                      // copy the data, ensure direction is correct
                                                      timer.reset(); // start the timer
                                                      if (view_device == DeviceSelector::DEVICE)
                                                      {
#ifdef DYNTUNE_DEBUG_ENABLED
                                                          std::cout
                                                            << "chain_" << i_chain << ": ker_" << i
                                                            << ":  OUTPUT -> host-2-device : view "
                                                            << jh << "->" << jd << std::endl;
#endif
                                                          Kokkos::deep_copy(view_d, view_h);
                                                      }
                                                      else
                                                      {
#ifdef DYNTUNE_DEBUG_ENABLED
                                                          std::cout
                                                            << "chain_" << i_chain << ": ker_" << i
                                                            << ":  OUTPUT -> device-2-host : view "
                                                            << jh << "->" << jd << std::endl;
#endif
                                                          Kokkos::deep_copy(view_h, view_d);
                                                      }
                                                      elapsed += timer.seconds();

                                                      // tick up our tracked output transfers
                                                      tracked_output_transfers++;
                                                  }
                                              }); // 5
                                        }
                                    }); // 4

                              } // 3
                          }
                      }); // 2

                } // 1

                // store the execution times in the vectors
                double chain_time = timer_all.seconds();
                chain_times[i_chain] += chain_time;
                chain_elapsed_times[i_chain] += elapsed;


                // execute the validation function (could be null)
#ifdef DYTUNE_DEBUG_ENABLED
                std::cout << "Finished chain " << i_chain << std::endl;
                std::cout << "Expected transfers (d2h), input: "
                          << chains_total_input_transfers_d2h[i_chain]
                          << " output: " << chains_total_output_transfers_d2h[i_chain] << std::endl;
                std::cout << "Expected transfers (h2d), input: "
                          << chains_total_input_transfers_h2d[i_chain]
                          << " output: " << chains_total_output_transfers_h2d[i_chain] << std::endl;
                std::cout << "Tracked transfers, input: " << tracked_input_transfers
                          << " output: " << tracked_output_transfers << std::endl;
#endif

                { // debug print
                    /*
                    bool success = true;
                    for (KernelSelector ksel : kernel_chain) {
                        for (size_t j : ksel.output_id) {
    //if (j > 4) continue;
                            iter_tuple(views_, [&]<typename ViewType>(size_t _j, ViewType& _views) {
    if (_j == j)
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

        } // 0

        total_operations_run++;

    } // end operator()

    void print_results(bool sort_results = true)
    {
        // this just goes through the kernels and prints the timing results
        std::cout << "==========================" << std::endl;
        std::cout << "===== Timing results =====" << std::endl << std::endl;

        std::cout << "Total number of chains run: " << kernel_chain_ids.size() << std::endl;
        std::cout << "Number of times run: " << total_operations_run << std::endl;
        std::cout << "Number of times each chain run: " << chain_runs << std::endl;
        std::cout << "Total number of times run: " << total_operations_run * chain_runs
                  << std::endl;


        std::cout << "Profiling results (avg): (chain_id, ops_time, total_time):" << std::endl;

        if (sort_results)
        {
            std::vector<size_t> sorted_ids(kernel_chains.size());
            std::iota(sorted_ids.begin(), sorted_ids.end(), 0);

            std::stable_sort(sorted_ids.begin(),
                             sorted_ids.end(),
                             [this](size_t i1, size_t i2)
                             { return this->chain_times[i1] < this->chain_times[i2]; });

            for (auto i_chain : sorted_ids)
            {
                double chain_time = chain_times[i_chain] / total_operations_run / chain_runs;
                double total_time =
                  chain_elapsed_times[i_chain] / total_operations_run / chain_runs;
                std::cout << "Chain " << std::setw(4) << i_chain << "\t" << std::scientific
                          << chain_time << "\t" << total_time << std::endl;
            }
        }
        else
        {
            for (size_t i_chain = 0; i_chain < kernel_chains.size(); i_chain++)
            {
                double chain_time = chain_times[i_chain] / total_operations_run / chain_runs;
                double total_time =
                  chain_elapsed_times[i_chain] / total_operations_run / chain_runs;
                std::cout << "Chain " << std::setw(4) << i_chain << "\t" << std::scientific
                          << chain_time << "\t" << total_time << std::endl;
            }
        }

        std::cout << std::endl << "==========================" << std::endl;
    }

}; // end Algorithm
