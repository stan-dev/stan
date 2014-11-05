#ifndef STAN__AGRAD__REV__ERROR_HANDLING__MATRIX__CHECK_POS_DEFINITE_HPP
#define STAN__AGRAD__REV__ERROR_HANDLING__MATRIX__CHECK_POS_DEFINITE_HPP

#include <stan/agrad/rev/numeric_limits.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/error_handling/matrix/check_pos_definite.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>

namespace stan {

  namespace agrad {

    template <typename T_result, class Policy>
    inline bool check_pos_definite(const std::string& function,
                                   const Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic>& y,
                                   const std::string& name,
                                   T_result* result,
                                   const Policy&) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<double,Dynamic,Dynamic> >::type size_type;
      Matrix<double,Dynamic,Dynamic> y_d(y.rows(),y.cols());
      for (size_type i = 0; i < y_d.rows(); i++) 
        for (size_type j = 0; j < y_d.cols(); j++)
          y_d(i,j) = y(i,j).val();
      return stan::error_handling::check_pos_definite(function,y_d,name,result,Policy());
    }
    
  }

}

#endif
