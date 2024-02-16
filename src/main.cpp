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
    private:
    std::vector<stdx::fixed_size_simd<_Tp, C>> data;

    public:
    Matrix(_Tp value) {
        std::vector<stdx::fixed_size_simd<_Tp, C>> vec;
        for (int r = 0; r < R; r++) {
            vec.push_back(stdx::fixed_size_simd<_Tp, C>(value));
        }
        this->data = vec;
    }

    Matrix(std::vector<stdx::fixed_size_simd<_Tp, C>> vec) {
        this->data = vec;
    }

    void print() {
        for (auto row: this->data) {
            print_simd(&row);
        }
    }

    template <int R2>
    Matrix<_Tp, R, R2> matmul(Matrix<_Tp, R2, C>& other) {
        std::vector<stdx::fixed_size_simd<_Tp, R2>> vec;
        for (int r1 = 0; r1 < R; r1++) { // R of ours
            _Tp* res = new _Tp[R2];
            for (int r2 = 0; r2 < R2; r2++) { // R of other
                stdx::fixed_size_simd<_Tp, C> tmp = this->data[r1] * other.data[r2];
                
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
    Matrix<int, 2, 3> a(1);
    Matrix<int, 2, 3> b(2);
    auto res = a.matmul(b);
    res.print();
}