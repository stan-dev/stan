#ifndef __STAN__DIFF__FWD__MATRIX__MDIVIDE_RIGHT_TRI_LOW_HPP__
#define __STAN__DIFF__FWD__MATRIX__MDIVIDE_RIGHT_TRI_LOW_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/diff/fwd/matrix/typedefs.hpp>
#include <stan/diff/fwd/matrix/to_fvar.hpp>
#include <stan/diff/fwd/matrix/multiply.hpp>
#include <stan/diff/fwd/matrix/inverse.hpp>
#include <stan/diff/fwd/fvar.hpp>

namespace stan {
  namespace diff {
    
    template<typename T1, typename T2,int R1, int C1, int R2, int C2>
    inline
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R2,C2> 
    mdivide_right_tri_low(const Eigen::Matrix<fvar<T1>, R1, C1>& m, 
                          const Eigen::Matrix<fvar<T2>, R2, C2>& n) {
      stan::math::validate_square(n, "mdivide_right_tri_low");
      stan::math::validate_multiplicable(m,n,"mdivide_right_tri_low");

      Eigen::Matrix<fvar<T2>,R2,C2> L(n.rows(),n.cols());
      L.setZero();

      for(size_type i = 0; i < n.rows(); i++) {
        for(size_type j = 0; (j < i + 1) && (j < n.cols()); j++)
          L(i,j) = n(i,j);
      }

      return stan::diff::multiply(m, stan::diff::inverse(L));
    }

    template<typename T1, typename T2,int R1, int C1, int R2, int C2>
    inline
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R2,C2> 
    mdivide_right_tri_low(const Eigen::Matrix<T1, R1, C1>& m, 
                          const Eigen::Matrix<fvar<T2>, R2, C2>& n) {
      stan::math::validate_square(n, "mdivide_right_tri_low");
      stan::math::validate_multiplicable(m,n,"mdivide_right_tri_low");

      Eigen::Matrix<fvar<T2>,R2,C2> L(n.rows(),n.cols());
      L.setZero();

      for(size_type i = 0; i < n.rows(); i++) {
        for(size_type j = 0; (j < i + 1) && (j < n.cols()); j++)
          L(i,j) = n(i,j);
      }

      return stan::diff::multiply(m, stan::diff::inverse(L));
    }

    template<typename T1, typename T2,int R1, int C1, int R2, int C2>
    inline
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>,R2,C2> 
    mdivide_right_tri_low(const Eigen::Matrix<fvar<T1>, R1, C1>& m, 
                          const Eigen::Matrix<T2, R2, C2>& n) {
      stan::math::validate_square(n, "mdivide_right_tri_low");
      stan::math::validate_multiplicable(m,n,"mdivide_right_tri_low");

      Eigen::Matrix<T2,R2,C2> L(n.rows(),n.cols());
      L.setZero();

      for(size_type i = 0; i < n.rows(); i++) {
        for(size_type j = 0; (j < i + 1) && (j < n.cols()); j++)
          L(i,j) = n(i,j);
      }

      return stan::diff::multiply(m, stan::diff::inverse(to_fvar(L)));
    }
  }
}
#endif
