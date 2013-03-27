#ifndef __STAN__MATH__MATRIX_ERROR_HANDLING_HPP__
#define __STAN__MATH__MATRIX_ERROR_HANDLING_HPP__

#include <stan/math/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/math/error_handling/matrix/check_size_match.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/error_handling/matrix/check_pos_definite.hpp>
#include <stan/math/error_handling/matrix/check_cov_matrix.hpp>
#include <stan/math/error_handling/matrix/check_corr_matrix.hpp>

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


    /**
     * Return <code>true</code> if the specified matrix is a valid
     * covariance matrix.  A valid covariance matrix must be symmetric
     * and positive definite.
     *
     * @param function
     * @param Sigma Matrix to test.
     * @param result
     * @return <code>true</code> if the matrix is a valid covariance matrix.
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings
    template <typename T_covar, typename T_result, class Policy>
    inline bool check_cov_matrix(const char* function,
         const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
         T_result* result,
         const Policy&) {
      if (!check_size_match(function, 
          Sigma.rows(), "Rows of covariance matrix",
          Sigma.cols(), "columns of covariance matrix",
                            result, Policy())) 
        return false;
      if (!check_positive(function, Sigma.rows(), "rows", result, Policy()))
        return false;
      if (!check_symmetric(function, Sigma, "Sigma", result, Policy()))
        return false;
      return true;
    }
    template <typename T_covar, typename T_result>
    inline bool check_cov_matrix(const char* function,
         const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
         T_result* result) {
      return check_cov_matrix(function,Sigma,result,default_policy());
      
    }
    template <typename T>
    inline bool check_cov_matrix(const char* function,
         const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
         T* result = 0) {
      return check_cov_matrix(function,Sigma,result,default_policy());
    }




    /**
     * Return <code>true</code> if the specified vector is unit vector.
     *
     * <p>The test that the values sum to 1 is done to within the
     * tolerance specified by <code>CONSTRAINT_TOLERANCE</code>.
     *
     * @param function
     * @param theta Vector to test.
     * @param name
     * @param result
     * @return <code>true</code> if the vector is a unit vector.
     */
    template <typename T_prob,
              typename T_result, 
              class Policy>
    bool check_unit_vector(const char* function,
                       const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta,
                       const char* name,
                       T_result* result,
                       const Policy&) {
      typedef typename Eigen::Matrix<T_prob,Eigen::Dynamic,1>::size_type size_t;
      using stan::math::policies::raise_domain_error;
      if (theta.size() == 0) {
        std::string message(name);
        message += " is not a valid unit vector. %1% elements in the vector.";
        T_result tmp = raise_domain_error<size_t, size_t>(function, 
                                                          message.c_str(), 
                                                          0, 
                                                          Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      T_prob ssq = theta.squaredNorm();
      if (fabs(1.0 - ssq) > CONSTRAINT_TOLERANCE) {
        std::stringstream msg;
        msg << "in function check_unit_vector(%1%), ";
        msg << name << " is not a valid unit vector.";
        msg << " The sum of the squares of the elements should be 1, but is " << ssq;
        T_result tmp = raise_domain_error<T_result,T_prob>(function, 
                                                           msg.str().c_str(), 
                                                           ssq, 
                                                           Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }                         
    template <typename T_y,
              typename T_result> // = typename T_prob_vector::value_type, 
    inline bool check_unit_vector(const char* function,
                              const Eigen::Matrix<T_y,Eigen::Dynamic,1>& theta,
                              const char* name,
                              T_result* result) {
      return check_unit_vector(function,theta,name,result,default_policy());
    }
    template <typename T>
    inline bool check_unit_vector(const char* function,
                              const Eigen::Matrix<T,Eigen::Dynamic,1>& theta,
                              const char* name,
                              T* result = 0) {
      return check_unit_vector(function,theta,name,result,default_policy());
    }




    /**
     * Return <code>true</code> if the specified vector is simplex.
     * To be a simplex, all values must be greater than or equal to 0
     * and the values must sum to 1.
     *
     * <p>The test that the values sum to 1 is done to within the
     * tolerance specified by <code>CONSTRAINT_TOLERANCE</code>.
     *
     * @param function
     * @param theta Vector to test.
     * @param name
     * @param result
     * @return <code>true</code> if the vector is a simplex.
     */
    template <typename T_prob,
              typename T_result, 
              class Policy>
    bool check_simplex(const char* function,
                       const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta,
                       const char* name,
                       T_result* result,
                       const Policy&) {
      typedef typename Eigen::Matrix<T_prob,Eigen::Dynamic,1>::size_type size_t;
      using stan::math::policies::raise_domain_error;
      if (theta.size() == 0) {
        std::string message(name);
        message += " is not a valid simplex. %1% elements in the vector.";
        T_result tmp = raise_domain_error<size_t, size_t>(function, 
                                                          message.c_str(), 
                                                          0, 
                                                          Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      if (fabs(1.0 - theta.sum()) > CONSTRAINT_TOLERANCE) {
        std::stringstream msg;
        T_prob sum = theta.sum();
        msg << "in function check_simplex(%1%), ";
        msg << name << " is not a valid simplex.";
        msg << " The sum of the elements should be 1, but is " << sum;
        T_result tmp = raise_domain_error<T_result,T_prob>(function, 
                                                           msg.str().c_str(), 
                                                           sum, 
                                                           Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      for (size_t n = 0; n < theta.size(); n++) {
        if (!(theta[n] >= 0)) {
          std::ostringstream stream;
          stream << name << " is not a valid simplex."
                 << " The element at " << n 
                 << " is %1%, but should be greater than or equal to 0";
          T_result tmp 
            = raise_domain_error<T_result,T_prob>(function, 
                                                  stream.str().c_str(), 
                                                  theta[n], 
                                                  Policy());
          if (result != 0)
            *result = tmp;
          return false;
        }
      }
      return true;
    }                         
    template <typename T_y,
              typename T_result> // = typename T_prob_vector::value_type, 
    inline bool check_simplex(const char* function,
                              const Eigen::Matrix<T_y,Eigen::Dynamic,1>& theta,
                              const char* name,
                              T_result* result) {
      return check_simplex(function,theta,name,result,default_policy());
    }
    template <typename T>
    inline bool check_simplex(const char* function,
                              const Eigen::Matrix<T,Eigen::Dynamic,1>& theta,
                              const char* name,
                              T* result = 0) {
      return check_simplex(function,theta,name,result,default_policy());
    }




    /**
     * Return <code>true</code> if the specified vector 
     * is sorted into increasing order.
     * There may be duplicate values.  Otherwise, raise a domain
     * error according to the specified policy.
     *
     * @param function
     * @param y Vector to test.
     * @param name
     * @param result
     * @tparam Policy Only the policy's type matters.
     * @return <code>true</code> if the vector has positive, ordered
     * values.
     */
    template <typename T_y, typename T_result, class Policy>
    bool check_ordered(const char* function,
                       const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                       const char* name,
                       T_result* result,
                       const Policy&) {
      using stan::math::policies::raise_domain_error;
      typedef typename Eigen::Matrix<T_y,Eigen::Dynamic,1>::size_type size_t;
      if (y.size() == 0) {
        return true;
      }
      for (size_t n = 1; n < y.size(); n++) {
        if (!(y[n] > y[n-1])) {
          std::ostringstream stream;
          stream << name << " is not a valid ordered vector."
                 << " The element at " << n 
                 << " is %1%, but should be greater than the previous element, "
                 << y[n-1];
          T_result tmp = raise_domain_error<T_result,T_y>(function, 
                                                          stream.str().c_str(), 
                                                          y[n], 
                                                          Policy());
          if (result != 0)
            *result = tmp;
          return false;
        }
      }
      return true;
    }                         
    template <typename T_y, typename T_result>
    bool check_ordered(const char* function,
                       const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                       const char* name,
                           T_result* result) {
      return check_ordered(function,y,name,result,default_policy());
    }
    template <typename T>
    bool check_ordered(const char* function,
                       const Eigen::Matrix<T,Eigen::Dynamic,1>& y,
                       const char* name,
                       T* result = 0) {
      return check_ordered(function,y,name,result,default_policy());
    }

    /**
     * Return <code>true</code> if the specified vector contains
     * only non-negative values and is sorted into increasing order.
     * There may be duplicate values.  Otherwise, raise a domain
     * error according to the specified policy.
     *
     * @param function
     * @param y Vector to test.
     * @param name
     * @param result
     * @tparam Policy Only the policy's type matters.
     * @return <code>true</code> if the vector has positive, ordered
     * values.
     */
    template <typename T_y, typename T_result, class Policy>
    bool check_positive_ordered(const char* function,
                                const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                                const char* name,
                                T_result* result,
                                const Policy&) {
      using stan::math::policies::raise_domain_error;
      typedef typename Eigen::Matrix<T_y,Eigen::Dynamic,1>::size_type size_t;
      if (y.size() == 0) {
        return true;
      }
      if (y[0] < 0) {
        std::ostringstream stream;
        stream << name << " is not a valid positive_ordered vector."
               << " The element at 0 is %1%, but should be postive.";
        T_result tmp = raise_domain_error<T_result,T_y>(function, 
                                                        stream.str().c_str(), 
                                                        y[0], 
                                                        Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      for (size_t n = 1; n < y.size(); n++) {
        if (!(y[n] > y[n-1])) {
          std::ostringstream stream;
          stream << name << " is not a valid ordered vector."
                 << " The element at " << n 
                 << " is %1%, but should be greater than the previous element, "
                 << y[n-1];
          T_result tmp = raise_domain_error<T_result,T_y>(function, 
                                                          stream.str().c_str(), 
                                                          y[n], 
                                                          Policy());
          if (result != 0)
            *result = tmp;
          return false;
        }
      }
      return true;
    }                         
    template <typename T_y, typename T_result>
    bool check_positive_ordered(const char* function,
                                const Eigen::Matrix<T_y,Eigen::Dynamic,1>& y,
                                const char* name,
                                T_result* result) {
      return check_positive_ordered(function,y,name,result,default_policy());
    }
    template <typename T>
    bool check_positive_ordered(const char* function,
                                const Eigen::Matrix<T,Eigen::Dynamic,1>& y,
                                const char* name,
                                T* result = 0) {
      return check_positive_ordered(function,y,name,result,default_policy());
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
