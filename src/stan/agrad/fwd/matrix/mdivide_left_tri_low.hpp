#ifndef __STAN__AGRAD__FWD__MATRIX__MDIVIDE_LEFT_TRI_LOW_HPP__
#define __STAN__AGRAD__FWD__MATRIX__MDIVIDE_LEFT_TRI_LOW_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/to_fvar.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/agrad/fwd/matrix/inverse.hpp>
#include <stan/agrad/fwd/fvar.hpp>

namespace stan {
  namespace agrad {
    
    template<typename T1, typename T2,int R1, int C1, int R2, int C2>
    inline
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C1> 
    mdivide_left_tri_low(const Eigen::Matrix<fvar<T1>, R1, C1>& m, 
                         const Eigen::Matrix<fvar<T2>, R2, C2>& n) {
      stan::math::validate_square(m, "mdivide_right_tri_low");
      stan::math::validate_multiplicable(m,n,"mdivide_right_tri_low");

      Eigen::Matrix<fvar<T1>,R1,C1> L(m.rows(),m.cols());
      L.setZero();

      for(size_type i = 0; i < m.rows(); i++) {
        for(size_type j = 0; (j < i + 1) && (j < m.cols()); j++)
          L(i,j) = m(i,j);
      }

      return stan::agrad::multiply(stan::agrad::inverse(L), n);
    }

    template<typename T1, typename T2,int R1, int C1, int R2, int C2>
    inline
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C1> 
    mdivide_left_tri_low(const Eigen::Matrix<T1, R1, C1>& m, 
                         const Eigen::Matrix<fvar<T2>, R2, C2>& n) {
      stan::math::validate_square(m, "mdivide_right_tri_low");
      stan::math::validate_multiplicable(m,n,"mdivide_right_tri_low");

      Eigen::Matrix<T1,R1,C1> L(m.rows(),m.cols());
      L.setZero();

      for(size_type i = 0; i < m.rows(); i++) {
        for(size_type j = 0; (j < i + 1) && (j < m.cols()); j++)
          L(i,j) = m(i,j);
      }

      return stan::agrad::multiply(stan::agrad::inverse(to_fvar(L)),n);
    }

    template<typename T1, typename T2,int R1, int C1, int R2, int C2>
    inline
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R1,C1> 
    mdivide_left_tri_low(const Eigen::Matrix<fvar<T1>, R1, C1>& m, 
                         const Eigen::Matrix<T2, R2, C2>& n) {
      stan::math::validate_square(m, "mdivide_right_tri_low");
      stan::math::validate_multiplicable(m,n,"mdivide_right_tri_low");

      Eigen::Matrix<fvar<T1>,R1,C1> L(m.rows(),m.cols());
      L.setZero();

      for(size_type i = 0; i < m.rows(); i++) {
        for(size_type j = 0; (j < i + 1) && (j < m.cols()); j++)
          L(i,j) = m(i,j);
      }

      return stan::agrad::multiply(stan::agrad::inverse(L),n);
    }
  }
}
#endif
