#ifndef STAN__AGRAD__FWD__MATRIX__COLUMNS__DOT_PRODUCT_HPP
#define STAN__AGRAD__FWD__MATRIX__COLUMNS__DOT_PRODUCT_HPP

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/error_handling/matrix/check_matching_dims.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/dot_product.hpp>
#include <stan/agrad/fwd/fvar.hpp>

namespace stan {
  namespace agrad {

    template<typename T, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<T>, 1, C1>
    columns_dot_product(const Eigen::Matrix<fvar<T>, R1, C1>& v1, 
                        const Eigen::Matrix<fvar<T>, R2, C2>& v2) {
      stan::error_handling::check_matching_dims("columns_dot_product", 
                                                "v1", v1,
                                                "v2", v2);
      Eigen::Matrix<fvar<T>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        Eigen::Matrix<fvar<T>,R1,C1> ccol1 = v1.col(j);
        Eigen::Matrix<fvar<T>,R2,C2> ccol2 = v2.col(j);
        ret(0,j) = dot_product(ccol1, ccol2);
      }
      return ret;
    }
    
    template<typename T, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<T>, 1, C1>
    columns_dot_product(const Eigen::Matrix<fvar<T>, R1, C1>& v1, 
                        const Eigen::Matrix<double, R2, C2>& v2) {
      stan::error_handling::check_matching_dims("columns_dot_product", 
                                                "v1", v1,
                                                "v2", v2);
      Eigen::Matrix<fvar<T>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        Eigen::Matrix<fvar<T>,R1,C1> ccol1 = v1.col(j);
        Eigen::Matrix<double,R2,C2> ccol = v2.col(j);
        ret(0,j) = dot_product(ccol1, ccol);
      }
      return ret;
    }

    template<typename T, int R1,int C1,int R2, int C2>
    inline 
    Eigen::Matrix<fvar<T>, 1, C1>
    columns_dot_product(const Eigen::Matrix<double, R1, C1>& v1, 
                        const Eigen::Matrix<fvar<T>, R2, C2>& v2) {
      stan::error_handling::check_matching_dims("columns_dot_product", 
                                                "v1", v1,
                                                "v2", v2);
      Eigen::Matrix<fvar<T>, 1, C1> ret(1,v1.cols());
      for (size_type j = 0; j < v1.cols(); ++j) {
        Eigen::Matrix<double,R1,C1> ccol = v1.col(j);
        Eigen::Matrix<fvar<T>,R2,C2> ccol2 = v2.col(j);
        ret(0,j) = dot_product(ccol, ccol2);
      }
      return ret;
    }
  }
}
#endif
