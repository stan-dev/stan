#ifndef STAN__AGRAD__REV__MATRIX__DETERMINANT_HPP
#define STAN__AGRAD__REV__MATRIX__DETERMINANT_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/error_handling/matrix/check_square.hpp>

// FIXME: use explicit files
#include <stan/agrad/rev.hpp> 

namespace stan {
  namespace agrad {

    namespace {
      template<int R,int C>
      class determinant_vari : public vari {
        int _rows;
        int _cols;
        double* A_;
        vari** _adjARef;
      public:
        determinant_vari(const Eigen::Matrix<var,R,C> &A)
          : vari(determinant_vari_calc(A)), 
            _rows(A.rows()),
            _cols(A.cols()),
            A_((double*)stan::agrad::memalloc_.alloc(sizeof(double) 
                                                     * A.rows() * A.cols())),
            _adjARef((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                          * A.rows() * A.cols()))
        {
          size_t pos = 0;
          for (size_type j = 0; j < _cols; j++) {
            for (size_type i = 0; i < _rows; i++) {
              A_[pos] = A(i,j).val();
              _adjARef[pos++] = A(i,j).vi_;
            }
          }
        }
        static 
        double determinant_vari_calc(const Eigen::Matrix<var,R,C> &A) {
          Eigen::Matrix<double,R,C> Ad(A.rows(),A.cols());
          for (size_type j = 0; j < A.rows(); j++)
            for (size_type i = 0; i < A.cols(); i++)
              Ad(i,j) = A(i,j).val();
          return Ad.determinant();
        }
        virtual void chain() {
          using Eigen::Matrix;
          using Eigen::Map;
          Matrix<double,R,C> adjA(_rows,_cols);
          adjA = (adj_ * val_) * 
            Map<Matrix<double,R,C> >(A_,_rows,_cols).inverse().transpose();
          size_t pos = 0;
          for (size_type j = 0; j < _cols; j++) {
            for (size_type i = 0; i < _rows; i++) {
              _adjARef[pos++]->adj_ += adjA(i,j);
            }
          }
        }
      };
    }

    template <int R, int C>
    inline var determinant(const Eigen::Matrix<var,R,C>& m) {
      stan::error_handling::check_square("determinant", "m", m);
      return var(new determinant_vari<R,C>(m));
    }
    
  }
}
#endif
