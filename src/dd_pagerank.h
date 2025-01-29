//
// Created by wortelus on 27.01.2025.
//

#ifndef DD_PAGERANK_H
#define DD_PAGERANK_H
#include <cstdint>
#include <cstddef>
#include <vector>


class DD_Pagerank
{
    int node_count;
    int edge_count;
    std::vector<int> outgoing_count;
    std::vector<std::vector<int>> outgoing_edges;
    std::vector<int> incoming_count;
    std::vector<std::vector<int>> incoming_edges;
    
    double d = 0.85;

    static size_t getFileSize(const char* fname);
    void init_pr(std::vector<double>& pr, bool parallel) const;
    static int pagerank_openmp_avx2(const std::vector<std::vector<int>>& incoming_edges,
                                    const std::vector<int>& outgoing_count,
                                    const std::vector<double>& pr, std::vector<double>& new_pr,
                                    int node_count,
                                    bool parallel, double d);
public:
    explicit DD_Pagerank(const char* filename);
    DD_Pagerank(const char* filename, bool parallel);
    std::vector<double> page_rank(size_t max_iter, bool parallel, int* out_iterations) const;
    void eval(std::vector<double> pr, int N = 5) const;
};


#endif //DD_PAGERANK_H
