#ifndef __STAN__AGRAD__MATRIX_ERROR_HANDLING_HPP__
#define __STAN__AGRAD__MATRIX_ERROR_HANDLING_HPP__

// global include
#include <stan/agrad/matrix.hpp>

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
