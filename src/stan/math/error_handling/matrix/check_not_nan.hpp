#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_NOT_NAN_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_NOT_NAN_HPP__

#include <sstream>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/default_policy.hpp>
#include <stan/math/error_handling/raise_domain_error.hpp>

namespace stan {
  namespace math {

    template <typename T_y, typename T_result, 
              int R, int C,
              class Policy>
    inline bool check_not_nan(const char* function,
                              const Eigen::Matrix<T_y,R,C>& y,
                              const char* name,
                              T_result* result,
                              const Policy&) {
      for (int i = 0; i < y.rows(); i++) {
        for (int j = 0; j < y.cols(); j++) {
          if (boost::math::isnan(y(i,j))) {
            std::ostringstream message;
            message << name << "[" << i << "," << j 
                    << "] is %1%, but must not be nan!";
            T_result tmp
              = policies::raise_domain_error<T_y>(function,
                                                  message.str().c_str(),
                                                  y(i,j), Policy());
            if (result != 0)
              *result = tmp;
            return false;
          }
        }
      }
      return true;
    }
    template <typename T_y, typename T_result,
              int R, int C>
    inline bool check_not_nan(const char* function,
                              const Eigen::Matrix<T_y,R,C>& y,
                              const char* name,
                              T_result* result) {
      return check_not_nan(function,y,name,result,default_policy());
    }
    template <typename T,
              int R, int C>
    inline bool check_not_nan(const char* function,
                              const Eigen::Matrix<T,R,C>& y,
                              const char* name,
                              T* result = 0) {
      return check_not_nan(function,y,name,result,default_policy());
    }

  }
}
#endif
