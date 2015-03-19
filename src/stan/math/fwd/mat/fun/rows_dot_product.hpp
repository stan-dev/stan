#ifndef STAN__MATH__FWD__MAT__FUN__ROWS_DOT_PRODUCT_HPP
#define STAN__MATH__FWD__MAT__FUN__ROWS_DOT_PRODUCT_HPP

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/err/check_matching_dims.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/dot_product.hpp>
#include <stan/math/fwd/core.hpp>


namespace stan {
  namespace agrad {

    template<typename T, int R1,int C1,int R2, int C2>
    inline
    Eigen::Matrix<fvar<T>, R1, 1>
    rows_dot_product(const Eigen::Matrix<fvar<T>, R1, C1>& v1,
                     const Eigen::Matrix<fvar<T>, R2, C2>& v2) {
      stan::math::check_matching_dims("rows_dot_product",
                                                "v1", v1,
                                                "v2", v2);
      Eigen::Matrix<fvar<T>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        Eigen::Matrix<fvar<T>,R1,C1> crow1 = v1.row(j);
        Eigen::Matrix<fvar<T>,R2,C2> crow2 = v2.row(j);
        ret(j,0) = dot_product(crow1, crow2);
      }
      return ret;
    }

    template<typename T, int R1,int C1,int R2, int C2>
    inline
    Eigen::Matrix<fvar<T>, R1, 1>
    rows_dot_product(const Eigen::Matrix<double, R1, C1>& v1,
                     const Eigen::Matrix<fvar<T>, R2, C2>& v2) {
      stan::math::check_matching_dims("rows_dot_product",
                                                "v1", v1,
                                                "v2", v2);
      Eigen::Matrix<fvar<T>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        Eigen::Matrix<double,R1,C1> crow = v1.row(j);
        Eigen::Matrix<fvar<T>,R2,C2> crow2 = v2.row(j);
        ret(j,0) = dot_product(crow, crow2);
      }
      return ret;
    }

    template<typename T, int R1,int C1,int R2, int C2>
    inline
    Eigen::Matrix<fvar<T>, R1, 1>
    rows_dot_product(const Eigen::Matrix<fvar<T>, R1, C1>& v1,
                     const Eigen::Matrix<double, R2, C2>& v2) {
      stan::math::check_matching_dims("rows_dot_product",
                                                "v1", v1,
                                                "v2", v2);
      Eigen::Matrix<fvar<T>, R1, 1> ret(v1.rows(),1);
      for (size_type j = 0; j < v1.rows(); ++j) {
        Eigen::Matrix<fvar<T>,R1,C1> crow1 = v1.row(j);
        Eigen::Matrix<double,R2,C2> crow = v2.row(j);
        ret(j,0) = dot_product(crow1, crow);
      }
      return ret;
    }
  }
}
#endif
