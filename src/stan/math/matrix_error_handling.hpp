#ifndef __STAN__MATH__MATRIX_ERROR_HANDLING_HPP__
#define __STAN__MATH__MATRIX_ERROR_HANDLING_HPP__

#include <limits>

#include <stan/math/boost_error_handling.hpp>
#include <stan/math/special_functions.hpp>

#include <stan/prob/transform.hpp>
#include <stan/math/matrix.hpp>

#include <boost/type_traits/make_unsigned.hpp>

namespace stan { 

  namespace math {

    template <typename T_y, typename T_result, class Policy>
    inline bool check_not_nan(const char* function,
                              const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                              const char* name,
                              T_result* result,
                              const Policy& /*pol*/) {
      using stan::math::policies::raise_domain_error;
      for (int i = 0; i < y.rows(); i++) {
        if (boost::math::isnan(y[i])) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must not be nan!";
          *result = raise_domain_error<T_result,T_y>(function,
                                                     message.str().c_str(),
                                                     y[i],
                                                     Policy());
          return false;
        }
      }
      return true;
    }


    template <typename T_y, typename T_result, class Policy>
    inline bool check_not_nan(const char* function,
                              const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                              const char* name,
                              T_result* result,
                              const Policy& /*pol*/) {
      for (int i = 0; i < y.rows(); i++) {
        for (int j = 0; j < y.cols(); j++) {
          if (boost::math::isnan(y(i,j))) {
            std::ostringstream message;
            message << name << "[" << i << "," << j << "] is %1%, but must not be nan!";
            *result = policies::raise_domain_error<T_y>(function,
                                              message.str().c_str(),
                                              y(i,j), Policy());
            return false;
          }
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
      using stan::math::policies::raise_domain_error;
      for (int i = 0; i < y.rows(); i++) {
        if (!boost::math::isfinite(y[i])) {
          std::ostringstream message;
          message << name << "[" << i << "] is %1%, but must be finite!";
          *result = raise_domain_error<T_result,T_y>(function,
                                                     message.str().c_str(),
                                                     y[i],
                                                     Policy());
          return false;
        }
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
        stream << "Sigma is not a valid covariance matrix. "
               << "Sigma must be symmetric and positive semi-definite. Sigma: \n" 
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

    template <typename T_covar, typename T_result, class Policy>
    inline bool check_corr_matrix(const char* function,
                                 const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
                                 T_result* result,
                                 const Policy& /*pol*/) {
      if (!stan::prob::corr_matrix_validate(Sigma)) {
        std::ostringstream stream;
        stream << "Sigma is not a valid correlation matrix."
               << "Sigma must be symmetric and positive semi-definite with ones on its diagonal. Sigma: \n" 
               << Sigma
               << "\nSigma(0,0): %1%";
        *result = policies::raise_domain_error<T_covar>(function,
                                              stream.str().c_str(), 
                                              Sigma(0,0),
                                              Policy());
        return false;
      }
      return true;
    }

    template <typename T_result, typename T_size1, typename T_size2, class Policy>
    inline bool check_size_match(const char* function,
                                 T_size1 i,
                                 T_size2 j,
                                 T_result* result,
                                 const Policy& /*pol*/) {
      using stan::math::policies::raise_domain_error;
      using boost::is_same; 
      if (is_same<T_size1, T_size2>::value) {
        if (i != j) {
          std::ostringstream msg;
          msg << "i and j must be same.  Found i=%1%, j=" << j;
          *result = raise_domain_error<T_result,T_size1>(function,
                                                         msg.str().c_str(),
                                                         i,
                                                         Policy());
          return false;
        }
      } else {
        using boost::make_unsigned;
        if ((typename make_unsigned<T_size1>::type)i != (typename make_unsigned<T_size2>::type)j) {
          std::ostringstream msg;
          msg << "i and j must be same.  Found i=%1%, j=" << j;
          *result = raise_domain_error<T_result,T_size1>(function,
                                                         msg.str().c_str(),
                                                         i,
                                                         Policy());
          return false;
        } 
      }
      return true;
    }


    template <typename T_prob, typename T_result, class Policy>
    inline bool check_simplex(const char* function,
                              const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta,
                              const char* name,
                              T_result* result,
                              const Policy& /*pol*/) {
      using stan::math::policies::raise_domain_error;
      if (!stan::prob::simplex_validate(theta)) {
        std::ostringstream stream;
        stream << name
               << "is not a valid simplex. The first element of the simplex is: %1%.";
        *result = raise_domain_error<T_result,T_prob>(function,
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
