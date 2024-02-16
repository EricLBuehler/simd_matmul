#include <chrono>
#include <experimental/simd>
#include <iostream>

namespace stdx = std::experimental;

template <typename _Tp, int R, int C>
class SlowMatrix {
   public:
    std::array<std::array<_Tp, C>, R>* data;

    SlowMatrix() {
        std::array<std::array<_Tp, C>, R>* arr =
            new std::array<std::array<_Tp, C>, R>{};
        this->data = arr;
    }

    SlowMatrix(_Tp value) {
        std::vector<std::vector<_Tp>>* vec =
            new std::vector<std::vector<_Tp>>{};
        for (int r = 0; r < R; r++) {
            vec.push_back(std::vector(C, value));
        }
        this->data = vec;
    }

    SlowMatrix(_Tp* inc_value) {
        std::array<std::array<_Tp, C>, R>* arr =
            new std::array<std::array<_Tp, C>, R>{};
        _Tp acc{};
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                acc += *inc_value;
                arr[r][c] = acc;
            }
        }
        this->data = arr;
    }

    SlowMatrix(std::array<std::array<_Tp, C>, R>* data) { this->data = data; }

    void print() const {
        for (auto row : this->data) {
            for (size_t i = 0; i < row.size(); i++) {
                std::cout << row[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    template <int _R, int _C>
    static SlowMatrix<_Tp, _C, _R> transpose(
        const SlowMatrix<_Tp, _R, _C>& matrix) {
        // Do the transpose
        std::array<std::array<_Tp, _R>, _C>* transpose =
            new std::array<std::array<_Tp, _R>, _C>{};

        int rows = matrix.data.size();
        int cols = matrix.data[0].size();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                transpose[j][i] = matrix.data[i][j];
            }
        }

        return Matrix<_Tp, _C, _R>(transpose);
    }

    template <int R2>
    SlowMatrix<_Tp, R, R2> matmul(const SlowMatrix<_Tp, C, R2>& other) {
        const SlowMatrix<_Tp, R2, C> other_T = transpose(other);
        return this->matmul_pre_T(other_T);
    }

    template <int R2>
    SlowMatrix<_Tp, R, R2> matmul_pre_T(const SlowMatrix<_Tp, R2, C>& other_T) {
        std::array<std::array<_Tp, R2>, R>* arr =
            new std::array<std::array<_Tp, R2>, R>{};
        for (int r1 = 0; r1 < R; r1++) {       // R of ours
            for (int r2 = 0; r2 < R2; r2++) {  // R of other
                // Do the mul here
                stdx::fixed_size_simd<_Tp, C> left;
                left.copy_from(&this->data[r1][0], stdx::element_aligned);

                stdx::fixed_size_simd<_Tp, C> right;
                right.copy_from(&other_T.data[r2][0], stdx::element_aligned);
                stdx::fixed_size_simd<_Tp, C> tmp = left * right;

                (*arr)[r1][r2] =
                    stdx::parallelism_v2::reduce(tmp, std::plus<>());
            }
        }
        return SlowMatrix<_Tp, R, R2>(arr);
    }

    ~SlowMatrix() { delete this->data; }
};

template <typename _Tp, int R, int C>
class Matrix {
   public:
    std::array<stdx::fixed_size_simd<_Tp, C>, R>* data;

    Matrix(SlowMatrix<_Tp, R, C>& from) {
        std::array<stdx::fixed_size_simd<_Tp, C>, R>* arr =
            new std::array<stdx::fixed_size_simd<_Tp, C>, R>{};
        for (typename std::array<std::array<_Tp, C>, R>::size_type i = 0;
             i < from.data->size(); i++) {
            stdx::fixed_size_simd<_Tp, C> row_simd;
            std::array<_Tp, C>& row = (*from.data)[i];
            row_simd.copy_from(&row[0], stdx::element_aligned);
            (*arr)[i] = row_simd;
        }
        this->data = arr;
    }

    template <int R2>
    SlowMatrix<_Tp, R, R2> matmul_pre_T(const Matrix<_Tp, R2, C>& other_T) {
        std::array<std::array<_Tp, R2>, R>* arr =
            new std::array<std::array<_Tp, R2>, R>{};

        for (int r1 = 0; r1 < R; r1++) {       // R of ours
            for (int r2 = 0; r2 < R2; r2++) {  // R of other
                // Do the mul here
                stdx::fixed_size_simd<_Tp, C> tmp =
                    (*this->data)[r1] * (*other_T.data)[r2];

                (*arr)[r1][r2] =
                    stdx::parallelism_v2::reduce(tmp, std::plus<>());
            }
        }
        return SlowMatrix<_Tp, R, R2>(arr);
    }

    void print() const {
        for (auto row : this->data) {
            for (size_t i = 0; i < row.size(); i++) {
                std::cout << row[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    ~Matrix() { delete this->data; }
};

template <size_t R, size_t C, size_t K>
std::array<std::array<int, K>, R> multiplyMatrices(
    const std::array<std::array<int, C>, R>* mat1,
    const std::array<std::array<int, K>, C>* mat2) {
    std::array<std::array<int, K>, R> result;

    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < K; ++j) {
            int sum = 0;
            for (size_t k = 0; k < C; ++k) {
                sum += (*mat1)[i][k] * (*mat2)[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

int main(int argc, char** argv) {
    constexpr int ROWS = 10;
    constexpr int COLS = 10;
    constexpr int WIDTH = 16;
    constexpr int TIMES = 10;

    SlowMatrix<int, ROWS, WIDTH> a_slow;
    SlowMatrix<int, COLS, WIDTH> b_slow;
    Matrix<int, ROWS, WIDTH> a(a_slow);
    Matrix<int, COLS, WIDTH> b(b_slow);

    int64_t total_time = 0;
    for (int i = 0; i < TIMES; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < 100; j++) {
            a.matmul_pre_T(b);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        total_time += duration.count();
    }
    std::cout << "SIMDATMUL Execution time: " << total_time / TIMES
              << " microseconds" << std::endl;

    std::array<std::array<int, WIDTH>, ROWS>* mat1 =
        new std::array<std::array<int, WIDTH>, ROWS>{};
    std::array<std::array<int, COLS>, WIDTH>* mat2 =
        new std::array<std::array<int, COLS>, WIDTH>{};

    int64_t total_time2 = 0;
    for (int i = 0; i < TIMES; i++) {
        auto start2 = std::chrono::high_resolution_clock::now();

        for (int j = 0; j < 100; j++) {
            multiplyMatrices(mat1, mat2);
        }

        auto end2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(
            end2 - start2);
        total_time2 += duration2.count();
    }
    std::cout << "Normal Execution time: " << total_time2 / TIMES
              << " microseconds" << std::endl;

    delete mat1;
    delete mat2;
}