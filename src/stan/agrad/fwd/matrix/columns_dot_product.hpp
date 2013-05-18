#ifndef __STAN__AGRAD__FWD__MATRIX__COLUMNS__DOT_PRODUCT_HPP__
#define __STAN__AGRAD__FWD__MATRIX__COLUMNS__DOT_PRODUCT_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_matching_dims.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/dot_product.hpp>
#include <stan/agrad/fwd/fvar.hpp>

namespace stan {
  namespace agrad {

    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1>
    columns_dot_product(const Eigen::Matrix<fvar<T1>, R1, C1>& v1, 
                        const Eigen::Matrix<fvar<T2>, R2, C2>& v2) {
      stan::math::validate_matching_dims(v1,v2,"columns_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        Eigen::Matrix<fvar<T1>,R1,C1> ccol1 = v1.col(j);
        Eigen::Matrix<fvar<T2>,R2,C2> ccol2 = v2.col(j);
        ret(0,j) = dot_product(ccol1, ccol2);
      }
      return ret;
    }
    
    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1>
    columns_dot_product(const Eigen::Matrix<fvar<T1>, R1, C1>& v1, 
                        const Eigen::Matrix<T2, R2, C2>& v2) {
      stan::math::validate_matching_dims(v1,v2,"columns_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        Eigen::Matrix<fvar<T1>,R1,C1> ccol1 = v1.col(j);
        Eigen::Matrix<T2,R2,C2> ccol = v2.col(j);
        Eigen::Matrix<fvar<T2>,R2,C2> ccol2 = to_fvar(ccol);
        ret(0,j) = dot_product(ccol1, ccol2);
      }
      return ret;
    }

    template<typename T1, typename T2, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1>
    columns_dot_product(const Eigen::Matrix<T1, R1, C1>& v1, 
                        const Eigen::Matrix<fvar<T2>, R2, C2>& v2) {
      stan::math::validate_matching_dims(v1,v2,"columns_dot_product");
      Eigen::Matrix<fvar<typename stan::return_type<T1,T2>::type>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        Eigen::Matrix<T1,R1,C1> ccol = v1.col(j);
        Eigen::Matrix<fvar<T1>,R1,C1> ccol1 = to_fvar(ccol);
        Eigen::Matrix<fvar<T2>,R2,C2> ccol2 = v2.col(j);
        ret(0,j) = dot_product(ccol1, ccol2);
      }
      return ret;
    }
  }
}
#endif
