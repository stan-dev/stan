#ifndef STAN__AGRAD__REV__ERROR_HANDLING__MATRIX__CHECK_POS_DEFINITE_HPP
#define STAN__AGRAD__REV__ERROR_HANDLING__MATRIX__CHECK_POS_DEFINITE_HPP

// global include
#include <stan/math/error_handling/matrix/check_pos_definite.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/numeric_limits.hpp>

namespace stan {
  namespace agrad {

    template <typename T_result, class Policy>
    inline bool check_pos_definite(const char* function,
                                   const Eigen::Matrix<var,Eigen::Dynamic,Eigen::Dynamic>& y,
                                   const char* name,
                                   T_result* result,
                                   const Policy&) {
        typedef 
        typename Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>::size_type 
        size_type;
        Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> y_d(y.rows(),y.cols());
        for (size_type i = 0; i < y_d.rows(); i++) 
          for (size_type j = 0; j < y_d.cols(); j++)
            y_d(i,j) = y(i,j).val();
        return stan::math::check_pos_definite(function,y_d,name,result,Policy());
    }
    
  }
}
#endif
