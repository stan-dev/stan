#ifndef __STAN__MATH__MATRIX_ERROR_HANDLING_HPP__
#define __STAN__MATH__MATRIX_ERROR_HANDLING_HPP__

#include <stan/math/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/error_handling/matrix/check_pos_definite.hpp>
#include <stan/math/error_handling/matrix/check_cov_matrix.hpp>
#include <stan/math/error_handling/matrix/check_corr_matrix.hpp>
#include <stan/math/error_handling/matrix/check_unit_vector.hpp>
#include <stan/math/error_handling/matrix/check_simplex.hpp>
#include <stan/math/error_handling/matrix/check_ordered.hpp>
#include <stan/math/error_handling/matrix/check_positive_ordered.hpp>

#include <sstream>
#include <stan/math/matrix.hpp>
#include <stan/meta/traits.hpp>
#include <stan/math/error_handling.hpp>
#include <boost/type_traits/common_type.hpp>

namespace stan { 

  namespace math {

    inline 
    void 
    validate_non_negative_index(const std::string& var_name,
                                const std::string& expr,
                                int val) {
      if (val < 0) {
        std::stringstream msg;
        msg << "Found negative dimension size in variable declaration"
            << "; variable=" << var_name
            << "; dimension size expression=" << expr
            << "; expression value=" << val;
        throw std::invalid_argument(msg.str());
      }
    }

    // error_handling functions for Eigen matrix

    template <typename T_y, typename T_result, class Policy>
    inline bool check_not_nan(const char* function,
                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
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
    template <typename T_y, typename T_result>
    inline bool check_not_nan(const char* function,
                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                  const char* name,
                  T_result* result) {
      return check_not_nan(function,y,name,result,default_policy());
    }
    template <typename T>
    inline bool check_not_nan(const char* function,
                  const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                  const char* name,
                  T* result = 0) {
      return check_not_nan(function,y,name,result,default_policy());
    }

  }
}

#endif
