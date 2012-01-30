#ifndef __STAN__MATHS__MATRIX_ERROR_HANDLING_HPP__
#define __STAN__MATHS__MATRIX_ERROR_HANDLING_HPP__

#include <limits>

#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>

#include <stan/maths/special_functions.hpp>

#include <stan/prob/transform.hpp>
#include <stan/maths/matrix.hpp>

#include <Eigen/Dense>


namespace stan { 

  namespace maths {

    template <typename T_y, typename T_result, class Policy>
    inline bool check_not_nan(const char* function,
                              const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                              const char* name,
                              T_result* result,
                              const Policy& /*pol*/) {
      for (int i = 0; i < y.rows(); i++) {
        if (boost::math::isnan(y[i])) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must not be nan!";
          *result = boost::math::policies::raise_domain_error<T_y>(function,
                                                                   message.str().c_str(),
                                                                   y[i], Policy());
          return false;
        }
      }
      return true;
    }



    template <typename T_y, typename T_result, class Policy>
    inline bool check_finite(const char* function,
                             const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                             const char* name,
                             T_result* result,
                             const Policy& /*pol*/) {
      for (int i = 0; i < y.rows(); i++) {
        if (!boost::math::isfinite(y[i])) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must be finite!";
          *result = boost::math::policies::raise_domain_error<T_y>(function,
                                                                   message.str().c_str(),
                                                                   y[i], Policy());
          return false;
        }
      }
      return true;
    }

    template <typename T_y, typename T_result, class Policy>
    inline bool check_finite(const char* function,
                             const typename 
                             stan::maths::EigenType<T_y>::vector& y,
                             const char* name,
                             T_result* result,
                             const Policy& /*pol*/) {
      using boost::math::policies::raise_domain_error;
      for (int i = 0; i < y.rows(); i++) {
        if (!boost::math::isfinite(y[i])) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must be finite!";
          *result = raise_domain_error<T_y>(function,
                                            message.str().c_str(),
                                            y[i], Policy());
          return false;
        }
      }
      return true;
    }

    template <typename T_covar, typename T_result, class Policy>
    inline bool check_cov_matrix(const char* function,
                                 const typename stan::maths::EigenType<T_covar>::matrix& Sigma,
                                 T_result* result,
                                 const Policy& /*pol*/) {
      using boost::math::policies::raise_domain_error;
      if (!stan::prob::cov_matrix_validate(Sigma)) {
        std::ostringstream stream;
        stream << "Sigma is not a valid covariance matrix."
               << " Sigma must be symmetric and positive semi-definite."
               << " Sigma:" << std::endl
               << Sigma << std::endl
               << "Sigma(0,0): %1%";
        *result = raise_domain_error<T_covar>(function,
                                              stream.str().c_str(), 
                                              Sigma(0,0),
                                              Policy());
        return false;
      }
      return true;
    }


    template <typename T_result, class Policy>
    inline bool check_size_match(const char* function,
                                 unsigned int i,
                                 unsigned int j,
                                 T_result* result,
                                 const Policy& /*pol*/) {
      if (i != j) {
        std::ostringstream msg;
        msg << "i and j must be same.  Found i=%1%, j=" << j;
        *result = boost::math::policies::raise_domain_error<double>(function,
                                                                    msg.str().c_str(),
                                                                    i,
                                                                    Policy());
        return false;
      }
      return true;
    }

    template <typename T_covar, typename T_result, class Policy>
    inline bool check_cov_matrix(const char* function,
                                 const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                                 T_result* result,
                                 const Policy& /*pol*/) {
      if (!stan::prob::cov_matrix_validate(Sigma)) {
        std::ostringstream stream;
        stream << "Sigma is not a valid covariance matrix. Sigma must be symmetric and positive semi-definite. Sigma: \n" 
               << Sigma
               << "\nSigma(0,0): %1%";
        *result = boost::math::policies::raise_domain_error<T_covar>(function,
                                                                     stream.str().c_str(), 
                                                                     Sigma(0,0),
                                                                     Policy());
        return false;
      }
      return true;
    }

    template <typename T_prob, typename T_result, class Policy>
    inline bool check_simplex(const char* function,
                              const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta,
                              const char* name,
                              T_result* result,
                              const Policy& /*pol*/) {
      if (!stan::prob::simplex_validate(theta)) {
        std::ostringstream stream;
        stream << name
               << "is not a valid simplex. The first element of the simplex is: %1%.";
        *result = boost::math::policies::raise_domain_error<T_prob>(function,
                                                                    stream.str().c_str(), 
                                                                    theta(0),
                                                                    Policy());
        return false;
      }
      return true;
    }

  }
}

#endif
