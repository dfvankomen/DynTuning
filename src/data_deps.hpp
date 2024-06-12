#include "common.hpp"
#include "kernel.hpp"

#include <iostream>
#include <unordered_map>
#include <vector>

class DataDependencyGraph
{
  public:
    DataDependencyGraph() {};
    ~DataDependencyGraph() {};

    struct Node
    {
        size_t view_id;
        std::vector<std::pair<size_t, Node*>> prev;
        std::vector<std::pair<size_t, Node*>> next;
        int indegree;
    };

    std::unordered_map<size_t, Node> graph;

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

    void add_edge(size_t start, size_t next, size_t kernel_id)
    {
        // create or get the start and next nodes
        Node* start_node = add_node(start);
        Node* next_node  = add_node(next);

        // add the next node to the list in start node and vice versa
        start_node->next.emplace_back(kernel_id, next_node);
        next_node->prev.emplace_back(kernel_id, start_node);
    }

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
