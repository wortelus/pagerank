//
// Created by wortelus on 27.01.2025.
//

#include "dd_pagerank.h"

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
#include <omp.h>

#include "consts.h"
#include "edge.h"

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
    this->edge_count = (int)edges.size();

    this->outgoing_count.resize(node_count, 0);
    this->incoming_count.resize(node_count, 0);
    this->outgoing_edges.resize(node_count);
    this->incoming_edges.resize(node_count);


    // Tímto to pravděpodobně moc nezoptimalizujeme, ale i tak v rámci demonstrace :-)
#pragma omp parallel if(parallel)
    {
#pragma omp for
        for (int i = 0; i < (int)edges.size(); i++)
        {
            int f = edges[i].from;
            int t = edges[i].to;

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

#pragma omp parallel if(parallel)
    {
#pragma omp for
        for (int i = 0; i < (int)edges.size(); i++)
        {
            int f = edges[i].from;
            int t = edges[i].to;

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
        int active_count = 0;
#pragma omp parallel for if(parallel) reduction(+:active_count)
        for (int u = 0; u < node_count; u++)
        {
            double sum = 0.0;
            for (auto v : incoming_edges[u])
            {
                double v_pr = pr[v];
                int v_outgoing_count = outgoing_count[v];
                sum += v_pr / v_outgoing_count;
            }

            new_pr[u] = (1.0 - this->d) / node_count + this->d * sum;

            if (std::abs(new_pr[u] - pr[u]) > EPSILON)
            {
                active_count++;
            }
        }

        // Pokud žádný uzel neprošel významnou změnou, skončíme
        if (active_count == 0) break;

        // Nastavení bool masky aktivních uzlů
#pragma omp parallel for if(parallel)
        for (int u = 0; u < node_count; u++)
        {
            active[u] = std::abs(new_pr[u] - pr[u]) > EPSILON;
        }

        // new_pr -> pr na konci iterace
#pragma omp parallel for if(parallel)
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

void DD_Pagerank::init_pr(std::vector<double>& pr, bool parallel) const
{
    // Paralelní inicializace 1/N
    double init_pr = 1.0 / static_cast<double>(node_count);
#pragma omp parallel for if(parallel)
    for (int i = 0; i < node_count; i++)
    {
        pr[i] = init_pr;
    }
}

void DD_Pagerank::eval(std::vector<double> pr, const int N) const
{
    std::vector<std::pair<size_t, double>> node_ranks;
    node_ranks.reserve(node_count);

    // Paralelní převod na vektor<index, pagerank>
#pragma omp parallel for
    for (int i = 0; i < node_count; i++)
    {
#pragma omp critical
        node_ranks.emplace_back(i, pr[i]);
    }

    // Seřazení podle PageRanku
    std::sort(node_ranks.begin(), node_ranks.end(), [](const auto& a, const auto& b)
    {
        return a.second > b.second;
    });

    // Výpis top N uzlů podle PageRanku
    printf("Top 5 nodes by PageRank:\n");
    for (int i = 0; i < std::min<size_t>(N, node_ranks.size()); i++)
    {
        printf("Node %zu: %.10f\n", node_ranks[i].first + 1, node_ranks[i].second);
    }
}
