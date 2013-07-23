#ifndef __STAN__AGRAD__REV__MATRIX__LOG_DETERMINANT_SPD_HPP__
#define __STAN__AGRAD__REV__MATRIX__LOG_DETERMINANT_SPD_HPP__

#include <vector>
#include <stan/math/error_handling/dom_err.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

// FIXME: use explicit files
#include <stan/agrad/agrad.hpp> 

namespace stan {
  namespace agrad {

    namespace {
      template <int R,int C>
      class log_determinant_spd_alloc : public chainable_alloc {
      public:
        virtual ~log_determinant_spd_alloc() {}
        
        Eigen::Matrix<double,R,C> _invA;
      };
      

      template<int R,int C>
      class log_determinant_spd_vari : public vari {
        log_determinant_spd_alloc<R,C> *_alloc;
        int _rows;
        int _cols;
        vari** _adjARef;
      public:
        log_determinant_spd_vari(const Eigen::Matrix<var,R,C> &A)
          : vari(log_determinant_spd_vari_calc(A,&_alloc)), 
            _rows(A.rows()),
            _cols(A.cols()),
            _adjARef((vari**)stan::agrad::memalloc_.alloc(sizeof(vari*) 
                                                          * A.rows() * A.cols()))
        {
          size_t pos = 0;
          for (size_type j = 0; j < _cols; j++) {
            for (size_type i = 0; i < _rows; i++) {
              _adjARef[pos++] = A(i,j).vi_;
            }
          }
        }

        static 
        double log_determinant_spd_vari_calc(const Eigen::Matrix<var,R,C> &A,
                                             log_determinant_spd_alloc<R,C> **alloc)
        {
          using stan::math::dom_err;

          // allocate space for information needed in chain
          *alloc = new log_determinant_spd_alloc<R,C>();

          // compute cholesky decomposition of A
          (*alloc)->_invA.resize(A.rows(),A.cols());
          for (size_type j = 0; j < A.cols(); j++)
            for (size_type i = 0; i < A.rows(); i++)
              (*alloc)->_invA(i,j) = A(i,j).val();
          Eigen::LDLT< Eigen::Matrix<double,R,C> > ldlt((*alloc)->_invA);
          if (ldlt.info() != Eigen::Success) {
            // Handle this better.
            (*alloc)->_invA.setZero(A.rows(),A.cols());
            double y = 0;
            double result = -std::numeric_limits<double>::infinity();
            return dom_err("log_determinant_spd(%1%)",
                           y,
                           "matrix argument",
                           "failed LDLT factorization","",
                           &result);
          }

          // compute the inverse of A (needed for the derivative)
          (*alloc)->_invA.setIdentity(A.rows(),A.cols());
          ldlt.solveInPlace((*alloc)->_invA);
          
          if (ldlt.isNegative() || (ldlt.vectorD().array() <= 1e-16).any()) {
            double y = 0;
            double result = -std::numeric_limits<double>::infinity();
            return dom_err("log_determinant_spd(%1%)",
                           y,
                           "matrix argument",
                           "matrix is negative definite","",
                           &result);
          }

          double ret = ldlt.vectorD().array().log().sum();
          if (!boost::math::isfinite(ret)) {
            double y = 0;
            double result = -std::numeric_limits<double>::infinity();
            return dom_err("log_determinant_spd(%1%)",
                           y,
                           "matrix argument",
                           "log determininant is infinite","",
                           &result);
          }
          return ret;
        }

        virtual void chain() {
          size_t pos = 0;
          for (size_type j = 0; j < _cols; j++) {
            for (size_type i = 0; i < _rows; i++) {
              _adjARef[pos++]->adj_ += adj_*_alloc->_invA(i,j);
            }
          }
        }
      };
    }

    template <int R, int C>
    inline var log_determinant_spd(const Eigen::Matrix<var,R,C>& m) {
      stan::math::validate_square(m,"log_determinant_spd");
      return var(new log_determinant_spd_vari<R,C>(m));
    }
    
  }
}
#endif
