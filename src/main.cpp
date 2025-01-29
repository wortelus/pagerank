#include <chrono>
#include <iostream>

#include "dd_pagerank.h"

void benchmark(const char* filename, int max_iterations, int benchmark_iterations) {
    DD_Pagerank dd_pagerank(filename);
    
    double total_time_parallel = 0.0;
    double total_time_serial = 0.0;
    int iterations = 0;

    for (int i = 0; i < benchmark_iterations; i++)
    {
        // Benchmark Paralelní PageRank
        auto start_parallel = std::chrono::high_resolution_clock::now();
        const auto pr_parallel = dd_pagerank.page_rank(max_iterations, true, &iterations);
        auto end_parallel = std::chrono::high_resolution_clock::now();
        total_time_parallel += std::chrono::duration<double>(end_parallel - start_parallel).count();
        
        // Benchmark Sériový PageRank
        auto start_serial = std::chrono::high_resolution_clock::now();
        const auto pr_serial = dd_pagerank.page_rank(max_iterations, false, &iterations);
        auto end_serial = std::chrono::high_resolution_clock::now();
        total_time_serial += std::chrono::duration<double>(end_serial - start_serial).count();
    }
    
    double avg_time_parallel = total_time_parallel / benchmark_iterations;
    double avg_time_serial = total_time_serial / benchmark_iterations;

    // Výsledky benchmarku
    std::cout << "Benchmark for " << filename << " with " << benchmark_iterations << " iterations." << std::endl;
    std::cout << "Parallel PageRank Avg Time: " << avg_time_parallel << "s." << std::endl;
    std::cout << "Serial PageRank Avg Time: " << avg_time_serial << "s." << std::endl;
    std::cout << "Speedup: " << avg_time_serial / avg_time_parallel << "-times." << std::endl;
}

void benchmark_loading(const char* filename, int benchmark_iterations) {
    double total_time_parallel = 0.0;
    double total_time_serial = 0.0;

    for (int i = 0; i < benchmark_iterations; i++)
    {
        auto start_parallel = std::chrono::high_resolution_clock::now();
        DD_Pagerank dd_pagerank_parallel(filename, true);
        auto end_parallel = std::chrono::high_resolution_clock::now();
        total_time_parallel += std::chrono::duration<double>(end_parallel - start_parallel).count();

        auto start_serial = std::chrono::high_resolution_clock::now();
        DD_Pagerank dd_pagerank_serial(filename, false);
        auto end_serial = std::chrono::high_resolution_clock::now();
        total_time_serial += std::chrono::duration<double>(end_serial - start_serial).count();
    }

    double avg_time_parallel = total_time_parallel / benchmark_iterations;
    double avg_time_serial = total_time_serial / benchmark_iterations;

    // Výsledky benchmarku
    std::cout << "Benchmark for " << filename << " with " << benchmark_iterations << " iterations." << std::endl;
    std::cout << "Parallel file loading Avg Time: " << avg_time_parallel << "s." << std::endl;
    std::cout << "Serial file loading Avg Time: " << avg_time_serial << "s." << std::endl;
    std::cout << "Speedup: " << avg_time_serial / avg_time_parallel << "-times." << std::endl;
}

int main(int argc, char* argv[]) {
    std::string filename = "web-BerkStan.txt"; // Default filename

    if (argc == 2) {
        filename = argv[1];
    }

    std::cout << "Parallel PageRank" << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();
    DD_Pagerank dd_pagerank(filename.c_str(), true);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::cout <<
    "Time (loading) of the parallel one: " <<
    std::chrono::duration<double>(end_time - start_time).count() << "s." << std::endl;

    int iterations = 0;
    start_time = std::chrono::high_resolution_clock::now();
    const auto pr = dd_pagerank.page_rank(1000, true, &iterations);
    end_time = std::chrono::high_resolution_clock::now();
    std::cout <<
        "Time (" << iterations << " iterations) of the parallel one: " <<
        std::chrono::duration<double>(end_time - start_time).count() << "s." << std::endl;
    dd_pagerank.eval(pr);
    

    std::cout << "Benchmark" << std::endl;
    benchmark("web-BerkStan.txt", 1000, 10);

    std::cout << "Benchmark Loading" << std::endl;
    benchmark_loading("web-BerkStan.txt", 3);
}
