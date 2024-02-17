#include <chrono>
#include <iostream>

#include "NumCpp.hpp"
#include "matrix.cpp"

template <size_t R, size_t C, size_t K>
std::unique_ptr<std::array<std::array<int, K>, R>> multiplyMatrices(
    const std::array<std::array<int, C>, R>* mat1,
    const std::array<std::array<int, K>, C>* mat2) {
    std::unique_ptr<std::array<std::array<int, K>, R>> out =
        std::make_unique<std::array<std::array<int, K>, R>>();
    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < K; ++j) {
            int sum = 0;
            for (size_t k = 0; k < C; ++k) {
                sum += (*mat1)[i][k] * (*mat2)[k][j];
            }
            (*out)[i][j] = sum;
        }
    }
    return out;
}

int main(int argc, char** argv) {
    constexpr int ROWS = 1000;
    constexpr int COLS = 1000;
    constexpr int WIDTH = 16;
    constexpr int TIMES = 100;
    constexpr int MUTLIPLIER = 1;

    SlowMatrix<int, ROWS, WIDTH> a_slow;
    SlowMatrix<int, COLS, WIDTH> b_slow;
    Matrix<int, ROWS, WIDTH> a(a_slow);
    Matrix<int, COLS, WIDTH> b(b_slow);

    int64_t total_time = 0;
    for (int i = 0; i < TIMES; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < MUTLIPLIER; j++) {
            a.matmul_pre_T(b);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count();
    }
    int64_t avg1 = total_time / TIMES;
    std::cout << "SIMDATMUL Execution time: " << avg1 << " microseconds"
              << std::endl;

    std::array<std::array<int, WIDTH>, ROWS>* mat1 =
        new std::array<std::array<int, WIDTH>, ROWS>{};
    std::array<std::array<int, COLS>, WIDTH>* mat2 =
        new std::array<std::array<int, COLS>, WIDTH>{};

    int64_t total_time2 = 0;
    for (int i = 0; i < TIMES; i++) {
        auto start2 = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < MUTLIPLIER; j++) {
            multiplyMatrices(mat1, mat2);
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(
            end2 - start2);
        total_time2 += duration2.count();
    }
    int64_t avg2 = total_time2 / TIMES;
    std::cout << "Normal Execution time: " << avg2 << " microseconds"
              << std::endl;

    std::cout << 1.0 - (((double)avg1) / ((double)avg2))
              << "% faster than Normal" << std::endl;

    nc::NdArray<int> nc1 = nc::zeros<int>((nc::uint32)ROWS, (nc::uint32)WIDTH);
    nc::NdArray<int> nc2 = nc::zeros<int>((nc::uint32)WIDTH, (nc::uint32)COLS);

    int64_t total_time3 = 0;
    for (int i = 0; i < TIMES; i++) {
        auto start3 = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < MUTLIPLIER; j++) {
            nc::matmul(nc1, nc2);
        }

        auto end3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::microseconds>(
            end3 - start3);
        total_time3 += duration3.count();
    }
    int64_t avg3 = total_time3 / TIMES;
    std::cout << "NumCpp Execution time: " << avg3 << " microseconds"
              << std::endl;

    std::cout << 1.0 - (((double)avg1) / ((double)avg3))
              << "% faster than NumCpp" << std::endl;

    // python3 -m timeit -c "import numpy;a=numpy.zeros([1000,16]);b=numpy.zeros([16,1000]);a@b"

    delete mat1;
    delete mat2;
}