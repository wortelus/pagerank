//
// Created by wortelus on 27.01.2025.
//

#include "dd_pagerank.h"
#include "consts.h"
#include "edge.h"

#include <omp.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iosfwd>
#include <sstream>
#include <string>
#include <vector>
#include <cwchar>
#include <iostream>

// Intel AVX2
#include <immintrin.h> // AVX2

///
/// @brief Sériový pro načtení grafu ze souboru
/// @param filename Cesta k souboru s grafem
///
DD_Pagerank::DD_Pagerank(const char* filename)
{
    std::ifstream file(filename);
    if (!file)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    std::cout << "Loading graph from " << filename << std::endl;

    // Přeskočte komentáře na začátku
    std::string line;
    int lines_skipped = 0;
    while (std::getline(file, line))
    {
        if (line.empty() || line[0] == '#')
        {
            lines_skipped++;
            continue;
        }
        break;
    }

    std::cout << "Skipped " << lines_skipped << " lines of comments." << std::endl;

    // Načítání hran (v 'line' již je první řádek s hranou)
    int from, to;
    int max_node = 0;
    std::vector<Edge> edges{};
    do
    {
        std::stringstream ss(line);
        ss >> from >> to;

        from -= 1;
        to -= 1;

        edges.push_back({from, to});
        if (from > max_node) max_node = from;
        if (to > max_node) max_node = to;
    }
    while (std::getline(file, line));

    // Dosáhli jsme EOF
    file.close();

    this->node_count = static_cast<int>(max_node) + 1;
    this->edge_count = static_cast<int>(edges.size());

    this->outgoing_count.resize(this->node_count, 0);
    this->outgoing_edges.resize(this->node_count);

    this->incoming_count.resize(this->node_count, 0);
    this->incoming_edges.resize(this->node_count);

    // this->outgoing_count
    // this->incoming_count
    // Výpočet počtu výstupních (vstupních) hran pro každý uzel
    for (const auto& [from, to] : edges)
    {
        this->outgoing_count[from]++;
        this->incoming_count[to]++;
    }

    // this->outgoing_edges
    // this->incoming_edges
    // Vytvoření předem alokovaného pole pro výstupní (vstupní) hrany
    for (auto i = 0; i < this->node_count; i++)
    {
        this->outgoing_edges[i].resize(this->outgoing_count[i]);
        this->incoming_edges[i].resize(this->incoming_count[i]);
    }

    // current_index - pomocný vektor pro sledování indexu vkládání do outgoing_edges
    std::vector<int> outgoing_index(this->node_count, 0);
    std::vector<int> incoming_index(this->node_count, 0);
    // Naplnění outgoing_edges target indexy uzlů
    for (const auto& [from, to] : edges)
    {
        int out_index = outgoing_index[from]++;
        this->outgoing_edges[from][out_index] = to;

        int in_index = incoming_index[to]++;
        this->incoming_edges[to][in_index] = from;
    }

    std::cout << "Graph loaded with " << this->node_count << " nodes and " << this->edge_count << " edges." <<
        std::endl;
}

///
/// @brief Získání velikosti souboru
/// @param fname Cesta k souboru
/// @return Velikost souboru v bytech
///
size_t DD_Pagerank::getFileSize(const char* fname)
{
    struct stat st{};
    if (stat(fname, &st) != 0)
    {
        perror("stat() failed");
        exit(EXIT_FAILURE);
    }
    return st.st_size;
}

///
/// @brief OpenMP paralelní/sériový konstruktor pro načtení grafu ze souboru
/// @param filename Cesta k souboru
/// @param parallel Paralelní zpracování (true) nebo sériové (false)
/// 
DD_Pagerank::DD_Pagerank(const char* filename, bool parallel)
{
    std::cout << "Loading graph from " << filename << std::endl;

    // Nejdřív otevřít soubor sériově a přeskočit komentáře
    // Soubor otevíráme binárně, jelikož budeme pracovat s offsety
    std::ifstream file_test(filename, std::ios::binary);
    if (!file_test)
    {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Preskočit komentáře na začátku souboru
    // A nastavit offset na začátek dat
    std::string line;
    int lines_skipped = 0;
    while (std::getline(file_test, line))
    {
        if (line.empty() || line[0] == '#')
        {
            lines_skipped++;
            continue;
        }
        // narazili jsme na první ne-komentář.. posuneme se o řádek zpět (rozdíl oproti implementaci výše)
        file_test.seekg(-static_cast<std::streamoff>(line.size() + 1), std::ios_base::cur);
        break;
    }
    // Zjistili jsme offset a zavřeme soubor
    std::streamoff start_offset = file_test.tellg();
    file_test.close();


    // Paralelní zpracování je dle chunků
    size_t file_size = getFileSize(filename);
    if (file_size <= 0)
    {
        std::cerr << "Error reading file while getting size." << std::endl;
        exit(EXIT_FAILURE);
    }

    int num_threads = parallel ? omp_get_max_threads() : 1;
    std::vector<std::vector<Edge>> thread_edges(num_threads);
    for (auto& v : thread_edges)
    {
        v.reserve(file_size / num_threads);
    }

    // Velikost chunku pro každé vlákno
    // + (num_threads - 1) pro zaokrouhlení nahoru, aby se pokryl celý soubor
    size_t chunk_size = (file_size - start_offset + num_threads - 1) / num_threads;

#pragma omp parallel if(parallel) default(none) shared(filename, file_size, start_offset, chunk_size, num_threads, thread_edges, std::cerr)
    {
        int tid = omp_get_thread_num();
        size_t start = start_offset + tid * chunk_size;
        size_t end = tid != num_threads - 1 ? std::min(start + chunk_size, file_size) : file_size;

        // Otevřít soubor v každém vlákně
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs)
        {
#pragma omp critical
            {
                std::cerr << "Thread " << tid << " cannot open file." << std::endl;
            }
#pragma omp cancel parallel
        }

        // Seek na začátek chunku
        ifs.seekg(static_cast<std::streamoff>(start));

        // Pokud nejsme na úplném začátku souboru
        // Pokud jsme uprostřed nebo na konci řádku, tak ho zahodíme (dočte ho předchozí na konci svého chunku)
        // Avšak pokud jsme na začátku řádku, tak ho zpracujeme (není součástí předchozího chunku)
        if (start != 0)
        {
            ifs.seekg(-1, std::ios::cur);
            char prev;
            ifs.get(prev);
            if (prev == '\n')
            {
                // std::cerr << "Previous char was '\\n', NOT skipping this line!" << std::endl;
            }
            else
            {
                std::string dummy;
                std::getline(ifs, dummy);
            }
        }

        // Načítání řádků a zpracovávání hran
        std::streampos current_pos = ifs.tellg();
        std::string l;
        while (current_pos >= 0 && current_pos < end && std::getline(ifs, l))
        {
            current_pos = ifs.tellg();

            // Komentáře uvnitř nejsou, takže je ignorujeme
            // if (l.empty() || l[0] == '#')
            // {
            //     continue;
            // }

            std::stringstream ss(l);
            int from, to;
            ss >> from >> to;
            // 1-based -> 0-based
            from -= 1;
            to -= 1;

            if (!ss.fail())
            {
                thread_edges[tid].push_back({from, to});
            }
            else
            {
                std::cerr << "Thread " << tid << " failed to parse line: " << l << std::endl;
            }
        }

        // Konec čtení souboru v rámci daného vlákna
        ifs.close();
    }
    //
    // konec paralelní sekce
    //

    // Sériová varianta (parallel == false) proběhne také v tom #pragma omp parallel, 
    // ale s num_threads == 1 ---> čili v jednom vlákně, pouze "wrapnutě"

    //
    // Sjednocení výsledků do jednoho vektoru
    //
    std::vector<Edge> edges;
    {
        size_t total = 0;
        for (auto& v : thread_edges)
        {
            total += v.size();
        }
        edges.reserve(total);
        for (auto& v : thread_edges)
        {
            edges.insert(edges.end(), v.begin(), v.end());
        }
    }

    // Najít max_node
    int max_node = 0;
    for (auto& e : edges)
    {
        if (e.from > max_node) max_node = e.from;
        if (e.to > max_node) max_node = e.to;
    }

    this->node_count = max_node + 1;
    this->edge_count = static_cast<int>(edges.size());

    this->outgoing_count.resize(node_count, 0);
    this->incoming_count.resize(node_count, 0);
    this->outgoing_edges.resize(node_count);
    this->incoming_edges.resize(node_count);


    // Tímto to pravděpodobně moc nezoptimalizujeme, ale i tak v rámci demonstrace :-)
#pragma omp parallel if(parallel) default(none) shared(edges, outgoing_count, incoming_count)
    {
#pragma omp for
        for (auto& edge : edges)
        {
            // ReSharper disable CppDFAUnusedValue
            // ReSharper disable CppDFAUnreadVariable
            int f = edge.from;
            int t = edge.to;
            // ReSharper restore CppDFAUnreadVariable
            // ReSharper restore CppDFAUnusedValue

#pragma omp atomic
            outgoing_count[f]++;
#pragma omp atomic
            incoming_count[t]++;
        }
    }

    // Alokace a defaultní hodnota pro adjacency list directed grafu
    for (int i = 0; i < node_count; i++)
    {
        outgoing_edges[i].resize(outgoing_count[i]);
        incoming_edges[i].resize(incoming_count[i]);
    }
    std::vector<int> out_idx(node_count, 0);
    std::vector<int> in_idx(node_count, 0);

#pragma omp parallel if(parallel) default(none) shared(edges, outgoing_edges, incoming_edges, out_idx, in_idx)
    {
#pragma omp for
        for (auto& edge : edges)
        {
            // ReSharper disable CppDFAUnreadVariable
            // ReSharper disable CppDFAUnusedValue
            int f = edge.from;
            int t = edge.to;
            // ReSharper restore CppDFAUnusedValue
            // ReSharper restore CppDFAUnreadVariable

            int pos_f, pos_t;
#pragma omp atomic capture
            {
                pos_f = out_idx[f];
                out_idx[f]++;
            }
#pragma omp atomic capture
            {
                pos_t = in_idx[t];
                in_idx[t]++;
            }
            outgoing_edges[f][pos_f] = t;
            incoming_edges[t][pos_t] = f;
        }
    }

    std::cout << "Graph loaded with " << this->node_count
        << " nodes and " << this->edge_count << " edges." << std::endl;
}

///
/// @brief OpenMP paralelní/sériový výpočet PageRanku
/// @param max_iter Maximální počet iterací
/// @param parallel Paralelní výpočet (true) nebo sériový (false)
/// @param out_iterations Počet iterací (výstup)
/// @return Vektor PageRanků (0-based)
std::vector<double> DD_Pagerank::page_rank(
    size_t max_iter, // Maximální počet iterací
    bool parallel, // true => OpenMP paralelně, false => sériově
    int* out_iterations // Počet iterací (výstup)
) const
{
    auto pr = std::vector<double>(node_count, 0.0);
    auto new_pr = std::vector<double>(node_count, 0.0);
    auto active = std::vector<bool>(node_count, true);

    // Inicializace PageRanku
    init_pr(pr, parallel);

    int iter = 1;
    while (iter <= max_iter)
    {
        // Výpočet PR(u)
        // Pokud parallel == true, bude použito #pragma omp parallel
        int active_count = pagerank_openmp_avx2(incoming_edges, outgoing_count, pr, new_pr, node_count, parallel, d);

        // Pokud žádný uzel neprošel významnou změnou, skončíme
        if (active_count == 0) break;

        // Nastavení bool masky aktivních uzlů
#pragma omp parallel for if(parallel) default(none) shared(active, pr, new_pr)
        for (int u = 0; u < node_count; u++)
        {
            active[u] = std::abs(new_pr[u] - pr[u]) > EPSILON;
        }

        // new_pr -> pr na konci iterace
#pragma omp parallel for if(parallel) default(none) shared(pr, new_pr)
        for (int u = 0; u < node_count; u++)
        {
            pr[u] = new_pr[u];
        }

        // Další iterace
        iter++;
    }

    if (out_iterations != nullptr)
        *out_iterations = iter;
    return pr;
}

int DD_Pagerank::pagerank_openmp_avx2(const std::vector<std::vector<int>>& incoming_edges,
                                      const std::vector<int>& outgoing_count,
                                      const std::vector<double>& pr,
                                      std::vector<double>& new_pr,
                                      int node_count,
                                      bool parallel,
                                      double d)
{
    int active_count = 0;

#pragma omp parallel for if(parallel) reduction(+:active_count) default(none) shared(node_count, pr, new_pr, incoming_edges, outgoing_count, d)
    for (int u = 0; u < node_count; u++)
    {
        size_t edge_count = incoming_edges[u].size();
        double sum = 0.0;

        // Intel AVX2 - Vynulování 256-bitového AVX2 registru (4x double)
        __m256d vec_sum = _mm256_setzero_pd();

        // Intel AVX2 Zpracování po čtveřicích
        int v = 0;
        for (; v + 3 < edge_count; v += 4)
        {
            __m256d v_pr = _mm256_set_pd(
                pr[incoming_edges[u][v + 3]], pr[incoming_edges[u][v + 2]],
                pr[incoming_edges[u][v + 1]], pr[incoming_edges[u][v]]);

            __m256d v_outgoing = _mm256_set_pd(
                outgoing_count[incoming_edges[u][v + 3]], outgoing_count[incoming_edges[u][v + 2]],
                outgoing_count[incoming_edges[u][v + 1]], outgoing_count[incoming_edges[u][v]]);

            __m256d div_result = _mm256_div_pd(v_pr, v_outgoing);

            // Clang-Tidy:
            // '_mm256_add_pd' is a non-portable x86_64 intrinsic function
#ifndef __AVX2__
#error "AVX2 support is required."
#endif
            vec_sum = _mm256_add_pd(vec_sum, div_result); // NOLINT(*-simd-intrinsics)
        }

        // 4 -> 2 -> 1 součet
        //
        // { a, b, c, d } → { a+b, c+d, a+b, c+d }
        vec_sum = _mm256_hadd_pd(vec_sum, vec_sum);

        // Clang-Tidy:
        // '_mm_add_pd' is a non-portable x86_64 intrinsic function
#ifndef __AVX2__
#error "AVX2 support is required."
#endif
        __m128d sum128 = _mm_add_pd( // NOLINT(*-simd-intrinsics)
            _mm256_extractf128_pd(vec_sum, 0), // { a+b, c+d }
            _mm256_extractf128_pd(vec_sum, 1) // { a+b, c+d }
        );
        sum += _mm_cvtsd_f64(sum128); // Finální součet

        // Zbytek (3 a méně) - sériově (non-AVX2)
        for (; v < edge_count; v++)
        {
            sum += pr[incoming_edges[u][v]] / outgoing_count[incoming_edges[u][v]];
        }

        // Nový PageRank
        new_pr[u] = (1.0 - d) / node_count + d * sum;

        // Přidání aktivního uzlu
        if (std::abs(new_pr[u] - pr[u]) > EPSILON)
        {
            active_count++;
        }
    }

    return active_count;
}

///
/// @brief OpenMP & AVX2 Inicializace PageRanku
/// @param pr Reference na vektor PageRanků
/// @param parallel Paralelní inicializace (true) nebo sériová (false)
/// 
void DD_Pagerank::init_pr(std::vector<double>& pr, bool parallel) const
{
    // Paralelní inicializace 1/N
    double init_pr = 1.0 / static_cast<double>(node_count);

#pragma omp parallel for if(parallel) default(none) shared(pr, init_pr)
    for (int i = 0; i < node_count; i++)
    {
#pragma omp simd
        for (int j = 0; j < 4; j++)
        {
            int index = i + j;
            if (index < node_count)
                pr[index] = init_pr;
        }
    }
}

///
/// @brief Výpis top N uzlů podle PageRanku
/// @param pr Vektor PageRanků
/// @param N Počet top uzlů
/// 
void DD_Pagerank::eval(std::vector<double> pr, const int N) const
{
    std::vector<std::pair<size_t, double>> node_ranks;
    node_ranks.reserve(node_count);

    // Převod na vektor<index, pagerank>
    // TODO: paralelizovat ?
    for (int i = 0; i < node_count; i++)
    {
        node_ranks.emplace_back(i, pr[i]);
    }

    // Seřazení podle PageRanku
    std::ranges::sort(node_ranks, [](const auto& a, const auto& b)
    {
        return a.second > b.second;
    });

    // Výpis top N uzlů podle PageRanku
    printf("Top 5 nodes by PageRank:\n");
    for (int i = 0; i < std::min<size_t>(N, node_ranks.size()); i++)
    {
        // 0-based -> 1-based (proto +1)
        printf("Node %zu: %.10f\n", node_ranks[i].first + 1, node_ranks[i].second);
    }
}
