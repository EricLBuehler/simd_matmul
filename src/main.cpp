#include <iostream>
#include <experimental/simd>

namespace stdx = std::experimental;

template <typename _Tp, int _Np>
void print_simd(stdx::fixed_size_simd<_Tp, _Np>* data) {
    for (size_t i = 0; i < data->size(); i++) {
        std::cout<<(*data)[i]<<" ";
    }
    std::cout<<std::endl;
}

template <typename _Tp, int R, int C>
class Matrix {
    public:
    std::vector<stdx::fixed_size_simd<_Tp, C>> data;

    Matrix(_Tp value) {
        std::vector<stdx::fixed_size_simd<_Tp, C>> vec;
        for (int r = 0; r < R; r++) {
            vec.push_back(stdx::fixed_size_simd<_Tp, C>(value));
        }
        this->data = vec;
    }

    Matrix(_Tp* inc_value) {
        std::vector<stdx::fixed_size_simd<_Tp, C>> vec;
        _Tp acc{};
        _Tp* acc_ptr = &acc;
        for (int r = 0; r < R; r++) {
            vec.push_back(stdx::fixed_size_simd<_Tp, C>([acc_ptr, inc_value](int _) { (*acc_ptr) += *inc_value; return *acc_ptr; }));
        }
        this->data = vec;
    }

    Matrix(std::vector<stdx::fixed_size_simd<_Tp, C>> vec) {
        this->data = vec;
    }

    void print() const {
        for (auto row: this->data) {
            print_simd(&row);
        }
    }

    
    template <int _R, int _C>
    static Matrix<_Tp, _C, _R> transpose(const Matrix<_Tp, _R, _C>& matrix) {
        std::vector<std::vector<_Tp>> vec; // .len=_R, [0].len=_C
        
        // Convert to vector
        for (int r = 0; r < _R; r++) {
            alignas(stdx::memory_alignment_v<stdx::fixed_size_simd<_Tp, _C>>)
            std::array<_Tp, stdx::fixed_size_simd<_Tp, _C>::size()> mem = {};
            matrix.data[r].copy_to(&mem[0], stdx::vector_aligned);
            vec.push_back(std::vector(mem.begin(), mem.end()));
        }

        // Do the transpose
        std::vector<std::vector<_Tp>> transpose(_C, std::vector<int>(_R));

        int rows = vec.size();
        int cols = vec[0].size();

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                transpose[j][i] = vec[i][j];
            }
        }

        // Back into simd
        std::vector<stdx::fixed_size_simd<_Tp, _R>> transpose_mat;
        for (std::vector<_Tp> row: transpose) {
            transpose_mat.push_back(stdx::fixed_size_simd<_Tp, _R>(row.data(), stdx::element_aligned));
        }     

        return Matrix<_Tp, _C, _R>(transpose_mat);
    }

    template <int R2>
    Matrix<_Tp, R, R2> matmul(const Matrix<_Tp, C, R2>& other) {
        const Matrix<_Tp, R2, C> other_T = transpose(other);
        std::vector<stdx::fixed_size_simd<_Tp, R2>> vec;
        for (int r1 = 0; r1 < R; r1++) { // R of ours
            _Tp* res = new _Tp[R2];
            for (int r2 = 0; r2 < R2; r2++) { // R of other
                stdx::fixed_size_simd<_Tp, C> tmp = this->data[r1] * other_T.data[r2];
                
                res[r2] = _Tp();
                for (std::size_t i = 0; i < tmp.size(); i++) {
                    const auto& data = tmp[i];
                    res[r2] += data;
                }
            }
            stdx::fixed_size_simd<_Tp, R2> simd(res, stdx::element_aligned);
            vec.push_back(simd);
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
    Matrix<int, 3, 4> b(&inc_val);
    std::cout<<"B:"<<std::endl;
    b.print();
    std::cout<<std::endl;
    auto res = a.matmul(b);
    res.print();
}