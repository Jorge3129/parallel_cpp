#include <iostream>
#include <vector>
#include <omp.h>
#include <utility>
#include <chrono>
#include <unordered_map>

struct PairHash {
    template <class T1, class T2>
    std::size_t operator() (const std::pair<T1, T2>& pair) const {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ hash2;
    }
};

int getIndex(int x, int y, int m_width) {
    return x + y * m_width;
}

double getNewValue(const std::vector<double>& matrix, int x, int y, int m_width, int n_height) {
    double sum = matrix[getIndex(x, y, m_width)];
    int count = 1;

    if (x > 0) {
        sum += matrix[getIndex(x - 1, y, m_width)];
        count++;
    }
    if (x < m_width - 1) {
        sum += matrix[getIndex(x + 1, y, m_width)];
        count++;
    }
    if (y > 0) {
        sum += matrix[getIndex(x, y - 1, m_width)];
        count++;
    }
    if (y < n_height - 1) {
        sum += matrix[getIndex(x, y + 1, m_width)];
        count++;
    }

    return sum / count;
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cout << "Expected args: <int width> <int height> <int processors> <long iterations>" << std::endl;
        return 1;
    }

    const int m_width = std::atoi(argv[1]);
    const int n_height = std::atoi(argv[2]);
    const int n_numOfProcessors = std::atoi(argv[3]);
    const long I_numIterations = std::atoi(argv[4]);

    // int I_numIterations, m_width, n_height, k_numFixedCells;

    // std::cin >> I_numIterations >> m_width >> n_height;
    std::vector<double> matrix(m_width * n_height, 0.0);
    std::unordered_map<std::pair<int, int>, double, PairHash> fixed_cells;

    for (int i = 0; i < std::min(m_width, n_height); ++i) {
        fixed_cells[{i, i}] = 1.0;
    }


    // std::cin >> k_numFixedCells;
    // for (int i = 0; i < k_numFixedCells; ++i) {
    //     int x, y;
    //     double V;
    //     std::cin >> x >> y >> V;
    //     fixed_cells.push_back({ x - 1, y - 1, V });
    // }

    omp_set_num_threads(n_numOfProcessors);

    for (const auto& cell : fixed_cells) {
        matrix[getIndex(cell.first.first, cell.first.second, m_width)] = cell.second;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < I_numIterations; ++iter) {
        std::vector<double> newMatrix = matrix;

#pragma omp parallel for collapse(2)
        for (int y = 0; y < n_height; ++y) {
            for (int x = 0; x < m_width; ++x) {
                auto it = fixed_cells.find({x, y});
                if (it != fixed_cells.end()) {
                    newMatrix[getIndex(x, y, m_width)] = it->second;
                }
                else {
                    newMatrix[getIndex(x, y, m_width)] = getNewValue(matrix, x, y, m_width, n_height);
                }
            }
        }

        matrix.swap(newMatrix);
    }

    auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed_time = end - start;

    if (n_height < 20 && m_width < 20) {
        for (int y = 0; y < n_height; ++y) {
            for (int x = 0; x < m_width; ++x) {
                std::cout << matrix[getIndex(x, y, m_width)] << " ";
            }
            std::cout << std::endl;
        }
    }

	std::cout << "Elapsed time: " << elapsed_time.count() << std::endl;

    return 0;
}
