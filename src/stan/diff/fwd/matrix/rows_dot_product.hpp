#ifndef __STAN__DIFF__FWD__MATRIX__ROWS__DOT_PRODUCT_HPP__
#define __STAN__DIFF__FWD__MATRIX__ROWS__DOT_PRODUCT_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_matching_dims.hpp>
#include <stan/diff/fwd/matrix/typedefs.hpp>
#include <stan/diff/fwd/matrix/dot_product.hpp>
#include <stan/diff/fwd/fvar.hpp>


namespace stan {
  namespace diff {

    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1>
    rows_dot_product(const Eigen::Matrix<fvar<T1>, R1, C1>& v1, 
                     const Eigen::Matrix<fvar<T2>, R2, C2>& v2) {
      stan::math::validate_matching_dims(v1,v2,"rows_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        Eigen::Matrix<fvar<T1>,R1,C1> crow1 = v1.row(j);
        Eigen::Matrix<fvar<T2>,R2,C2> crow2 = v2.row(j);
        ret(j,0) = dot_product(crow1, crow2);
      }
      return ret;
    }
    
    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1>
    rows_dot_product(const Eigen::Matrix<T1, R1, C1>& v1, 
                     const Eigen::Matrix<fvar<T2>, R2, C2>& v2) {
      stan::math::validate_matching_dims(v1,v2,"rows_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        Eigen::Matrix<T1,R1,C1> crow = v1.row(j);
        Eigen::Matrix<fvar<T1>,R1,C1> crow1 = to_fvar(crow);
        Eigen::Matrix<fvar<T2>,R2,C2> crow2 = v2.row(j);
        ret(j,0) = dot_product(crow1, crow2);
      }
      return ret;
    }

    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1>
    rows_dot_product(const Eigen::Matrix<fvar<T1>, R1, C1>& v1, 
                     const Eigen::Matrix<T2, R2, C2>& v2) {
      stan::math::validate_matching_dims(v1,v2,"rows_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        Eigen::Matrix<fvar<T1>,R1,C1> crow1 = v1.row(j);
        Eigen::Matrix<T2,R2,C2> crow = v2.row(j);
        Eigen::Matrix<fvar<T2>,R2,C2> crow2 = to_fvar(crow);
        ret(j,0) = dot_product(crow1, crow2);
      }
      return ret;
    }
  }
}
#endif
