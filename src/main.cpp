#include <iostream>
#include <experimental/simd>

namespace stdx = std::experimental;

template <typename _Tp, int R, int C>
class Matrix {
    public:
    std::vector<std::vector<_Tp>> data;

    Matrix(_Tp value) {
        std::vector<std::vector<_Tp>> vec;
        for (int r = 0; r < R; r++) {
            vec.push_back(std::vector(C, value));
        }
        this->data = vec;
    }

    Matrix(_Tp* inc_value) {
        std::vector<std::vector<_Tp>> vec;
        _Tp acc{};
        for (int r = 0; r < R; r++) {
            std::vector<_Tp> inner;
            for (int c = 0; c < C; c++) {
                acc += *inc_value;
                inner.push_back(acc);
            }
            vec.push_back(inner);
        }
        this->data = vec;
    }

    Matrix(std::vector<std::vector<_Tp>> vec) {
        this->data = vec;
    }

    void print() const {
        for (auto row: this->data) {
            for (size_t i = 0; i < row.size(); i++) {
                std::cout<<row[i]<<" ";
            }
            std::cout<<std::endl;
        }
    }

    
    template <int _R, int _C>
    static Matrix<_Tp, _C, _R> transpose(const Matrix<_Tp, _R, _C>& matrix) {
        // Do the transpose
        std::vector<std::vector<_Tp>> transpose(_C, std::vector<int>(_R));

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
        std::vector<std::vector<_Tp>> vec;
        for (int r1 = 0; r1 < R; r1++) { // R of ours
            _Tp* res = new _Tp[R2];
            for (int r2 = 0; r2 < R2; r2++) { // R of other
                // Do the mul here
                stdx::fixed_size_simd<_Tp, C> tmp = stdx::fixed_size_simd<_Tp, C>(this->data[r1].data(), stdx::element_aligned) * stdx::fixed_size_simd<_Tp, C>(other_T.data[r2].data(), stdx::element_aligned);
                
                /// Sum
                res[r2] = _Tp(); // Default
                for (std::size_t i = 0; i < tmp.size(); i++) {
                    const auto& data = tmp[i];
                    res[r2] += data;
                }
            }
            vec.push_back(std::vector<_Tp>(res, res+R2));
        }
        return Matrix<_Tp, R, R2>(vec);
    }
};

int main(int argc, char **argv) {
    int inc_val = 1;
    Matrix<int, 2, 3> a(&inc_val);
    std::cout<<"A:"<<std::endl;
    a.print();
    std::cout<<std::endl;
    Matrix<int, 4, 3> b(&inc_val);
    std::cout<<"B:"<<std::endl;
    b.print();
    std::cout<<std::endl;
    auto res = a.matmul_pre_T(b);
    res.print();
}