#include "common.hpp"
#include "kernel.hpp"

#include <iostream>
#include <unordered_map>
#include <vector>

/**
 * @brief The data dependency graph class
 *
 * This class serves to help identify any data dependencies. It also stores
 * global IDs for each data used. The graph's intention is to help the optimizer
 * identify what dependencies exist so that it can transfer data to the proper device
 * if or when necessary. This shouldn't necessarily be used anywhere else.
 *
 */
class DataDependencyGraph
{
  public:
    DataDependencyGraph() {};
    ~DataDependencyGraph() {};

    /**
     * @brief Simple structure for the Node object that is stored
     *
     */
    struct Node
    {
        // Global ID of the view
        size_t view_id;
        // Vector storing everything the view points from
        std::vector<std::pair<size_t, Node*>> prev;
        // Vector storing everything that the view points to
        std::vector<std::pair<size_t, Node*>> next;
        // The in-degree of the node
        int indegree;
    };

    // The graph object itself is just an unordered map
    std::unordered_map<size_t, Node> graph;

    /**
     * @brief Add a node to the data dependency graph
     *
     * @param view_id_in Global ID for the input data
     * @return Node* Reference to stored node
     */
    Node* add_node(size_t view_id_in)
    {
        if (auto found_node = graph.find(view_id_in); found_node != graph.end())
        {
            // when adding a node, we need to see if it exists first, if it does then we return the
            // pointer to the node as stored
            return &found_node->second;
        }
        else
        {
            // otherwise, we add it with the unordered map and initialize everything to empty stuff
            graph[view_id_in] = Node({ view_id_in, {}, {}, 0 });
            // then return the nemwly created node
            return &graph[view_id_in];
        }
    }

    /**
     * @brief Find a node based on id
     *
     * @param view_id_in Node ID to be searched for
     * @return Node* Reference to the desired node, nullptr if not found
     */
    Node* find_node(size_t view_id_in)
    {
        if (auto found_node = graph.find(view_id_in); found_node != graph.end())
        {
            return &found_node->second;
        }
        else
        {
            return nullptr;
        }
    }

    /**
     * @brief Add an edge to the data dependency graph
     *
     * @param start Node ID for start of edge
     * @param next Node ID for where edge points to
     * @param kernel_id Kernel ID used for storage
     */
    void add_edge(size_t start, size_t next, size_t kernel_id)
    {
        // create or get the start and next nodes
        Node* start_node = add_node(start);
        Node* next_node  = add_node(next);

        // add the next node to the list in start node and vice versa
        start_node->next.emplace_back(kernel_id, next_node);
        next_node->prev.emplace_back(kernel_id, start_node);
    }

    /**
     * @brief Helper function that prints graphs for debugging
     *
     */
    void print_graph_normal()
    {
        // simple printing here
        for (auto it = graph.cbegin(); it != graph.cend(); it++)
        {
            std::cout << "data id: " << it->first << " | " << it->second.view_id << std::endl
                      << "  In edges:\t";
            for (size_t ii = 0; ii < it->second.prev.size(); ii++)
            {
                std::cout << it->second.prev[ii].second->view_id << "-k("
                          << it->second.prev[ii].first << ") ";
            }
            std::cout << std::endl << "  Out edges:\t";
            for (size_t ii = 0; ii < it->second.next.size(); ii++)
            {
                std::cout << it->second.next[ii].second->view_id << "-k("
                          << it->second.next[ii].first << ") ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};
