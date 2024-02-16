#include <chrono>
#include <experimental/simd>
#include <iostream>

namespace stdx = std::experimental;

template <typename _Tp, int R, int C>
class Matrix {
   public:
    std::array<std::array<_Tp, C>, R> data;

    Matrix() {
        std::array<std::array<_Tp, C>, R> arr{};
        this->data = arr;
    }

    Matrix(_Tp value) {
        std::vector<std::vector<_Tp>> vec;
        for (int r = 0; r < R; r++) {
            vec.push_back(std::vector(C, value));
        }
        this->data = vec;
    }

    Matrix(_Tp* inc_value) {
        std::array<std::array<_Tp, C>, R> arr;
        _Tp acc{};
        for (int r = 0; r < R; r++) {
            for (int c = 0; c < C; c++) {
                acc += *inc_value;
                arr[r][c] = acc;
            }
        }
        this->data = arr;
    }

    Matrix(std::array<std::array<_Tp, C>, R> data) { this->data = data; }

    void print() const {
        for (auto row : this->data) {
            for (size_t i = 0; i < row.size(); i++) {
                std::cout << row[i] << " ";
            }
            std::cout << std::endl;
        }
    }

    template <int _R, int _C>
    static Matrix<_Tp, _C, _R> transpose(const Matrix<_Tp, _R, _C>& matrix) {
        // Do the transpose
        std::array<std::array<_Tp, _R>, _C> transpose;

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
    Matrix<_Tp, R, R2> matmul(const Matrix<_Tp, C, R2>& other) {
        const Matrix<_Tp, R2, C> other_T = transpose(other);
        return this->matmul_pre_T(other_T);
    }

    template <int R2>
    Matrix<_Tp, R, R2> matmul_pre_T(const Matrix<_Tp, R2, C>& other_T) {
        std::array<std::array<_Tp, R2>, R> arr;
        for (int r1 = 0; r1 < R; r1++) {       // R of ours
            for (int r2 = 0; r2 < R2; r2++) {  // R of other
                // Do the mul here
                stdx::fixed_size_simd<_Tp, C> tmp =
                    stdx::fixed_size_simd<_Tp, C>(this->data[r1].data(),
                                                  stdx::element_aligned) *
                    stdx::fixed_size_simd<_Tp, C>(other_T.data[r2].data(),
                                                  stdx::element_aligned);

                /// Sum
                _Tp acc{};  // Default
                for (std::size_t i = 0; i < tmp.size(); i++) {
                    const auto& data = tmp[i];
                    acc += data;
                }
                arr[r1][r2] = acc;
            }
        }
        return Matrix<_Tp, R, R2>(arr);
    }
};

template <size_t R, size_t C, size_t K>
std::array<std::array<int, K>, R> multiplyMatrices(
    const std::array<std::array<int, C>, R>& mat1,
    const std::array<std::array<int, K>, C>& mat2) {
    std::array<std::array<int, K>, R> result;

    for (size_t i = 0; i < R; ++i) {
        for (size_t j = 0; j < K; ++j) {
            int sum = 0;
            for (size_t k = 0; k < C; ++k) {
                sum += mat1[i][k] * mat2[k][j];
            }
            result[i][j] = sum;
        }
    }

    return result;
}

int main(int argc, char** argv) {
    Matrix<int, 2000, 3> a;
    Matrix<int, 40, 3> b;
    auto start = std::chrono::high_resolution_clock::now();

    auto res = a.matmul_pre_T(b);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "SIMDATMUL Execution time: " << duration.count()
              << " microseconds" << std::endl;

    std::array<std::array<int, 3>, 2000> mat1;
    std::array<std::array<int, 40>, 3> mat2;
    auto start2 = std::chrono::high_resolution_clock::now();

    auto res2 = multiplyMatrices(mat1, mat2);

    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 =
        std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2);
    std::cout << "Normal Execution time: " << duration2.count()
              << " microseconds" << std::endl;
}