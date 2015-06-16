#ifndef STAN_MATH_REV_MAT_FUN_DETERMINANT_HPP
#define STAN_MATH_REV_MAT_FUN_DETERMINANT_HPP

#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <vector>

namespace stan {
  namespace math {

    namespace {
      template<int R, int C>
      class determinant_vari : public vari {
        int _rows;
        int _cols;
        double* A_;
        vari** _adjARef;

      public:
        explicit determinant_vari(const Eigen::Matrix<var, R, C> &A)
          : vari(determinant_vari_calc(A)),
            _rows(A.rows()),
            _cols(A.cols()),
            A_(reinterpret_cast<double*>
               (stan::math::ChainableStack::memalloc_
                .alloc(sizeof(double) * A.rows() * A.cols()))),
            _adjARef(reinterpret_cast<vari**>
                     (stan::math::ChainableStack::memalloc_
                      .alloc(sizeof(vari*) * A.rows() * A.cols()))) {
          size_t pos = 0;
          for (size_type j = 0; j < _cols; j++) {
            for (size_type i = 0; i < _rows; i++) {
              A_[pos] = A(i, j).val();
              _adjARef[pos++] = A(i, j).vi_;
            }
          }
        }
        static
        double determinant_vari_calc(const Eigen::Matrix<var, R, C> &A) {
          Eigen::Matrix<double, R, C> Ad(A.rows(), A.cols());
          for (size_type j = 0; j < A.rows(); j++)
            for (size_type i = 0; i < A.cols(); i++)
              Ad(i, j) = A(i, j).val();
          return Ad.determinant();
        }
        virtual void chain() {
          using Eigen::Matrix;
          using Eigen::Map;
          Matrix<double, R, C> adjA(_rows, _cols);
          adjA = (adj_ * val_) *
            Map<Matrix<double, R, C> >(A_, _rows, _cols).inverse().transpose();
          size_t pos = 0;
          for (size_type j = 0; j < _cols; j++) {
            for (size_type i = 0; i < _rows; i++) {
              _adjARef[pos++]->adj_ += adjA(i, j);
            }
          }
        }
      };
    }

    template <int R, int C>
    inline var determinant(const Eigen::Matrix<var, R, C>& m) {
      stan::math::check_square("determinant", "m", m);
      return var(new determinant_vari<R, C>(m));
    }

  }
}
#endif
