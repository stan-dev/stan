#ifndef STAN__MATH__REV__MAT__ERR__CHECK_POS_DEFINITE_HPP
#define STAN__MATH__REV__MAT__ERR__CHECK_POS_DEFINITE_HPP

#include <stan/math/prim/mat/err/check_pos_definite.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/core/numeric_limits.hpp>

namespace stan {

  namespace agrad {

    inline bool check_pos_definite(const char* function,
                                   const char* name,
                                   const Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic>& y) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef index_type<Matrix<double,Dynamic,Dynamic> >::type size_type;
      Matrix<double,Dynamic,Dynamic> y_d(y.rows(),y.cols());
      for (size_type i = 0; i < y_d.rows(); i++) 
        for (size_type j = 0; j < y_d.cols(); j++)
          y_d(i,j) = y(i,j).val();
      return stan::math::check_pos_definite(function, name, y_d);
    }
    
  }

}

#endif
