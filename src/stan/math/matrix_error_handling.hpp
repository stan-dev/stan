#ifndef __STAN__MATH__MATRIX_ERROR_HANDLING_HPP__
#define __STAN__MATH__MATRIX_ERROR_HANDLING_HPP__

#include <stan/meta/traits.hpp>
#include <stan/math/error_handling.hpp>

#include <boost/type_traits/common_type.hpp>

namespace stan { 

  namespace math {
    
    // FIXME: update warnings
    template <typename T_size1, typename T_size2, typename T_result,
              class Policy>
    inline bool check_size_match(const char* function,
                                 T_size1 i,
                                 T_size2 j,
                                 T_result* result,
                                 const Policy&) {
      using stan::math::policies::raise_domain_error;
      typedef typename boost::common_type<T_size1,T_size2>::type common_type;
      if (static_cast<common_type>(i) != static_cast<common_type>(j)) {
        std::ostringstream msg;
        msg << "i and j must be same. Found i=%1%, j=" << j;
        T_result tmp = raise_domain_error<T_result,T_size1>(function,
                                                            msg.str().c_str(),
                                                            i,
                                                            Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }

    template <typename T_size1, typename T_size2, typename T_result>
    inline
    bool check_size_match(const char* function,
                          T_size1 i,
                          T_size2 j,
                          T_result* result) {
      return check_size_match(function,i,j,result,default_policy());
    }

    template <typename T_size1, typename T_size2>
    inline
    bool check_size_match(const char* function,
                          T_size1 i,
                          T_size2 j,
                          T_size1* result = 0) {
      return check_size_match(function,i,j,result,default_policy());
    }
    


    /**
     * Return <code>true</code> if the specified matrix is symmetric
     * 
     * NOTE: squareness is not checked by this function
     *
     * @param function 
     * @param y Matrix to test.
     * @param name
     * @param result
     * @return <code>true</code> if the matrix is symmetric.
     * @tparam T Type of scalar.
     */
    template <typename T_y, typename T_result, class Policy>
    inline bool check_symmetric(const char* function,
                const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const char* name,
                T_result* result,
                const Policy&) {
      typedef 
        typename Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>::size_type 
        size_type;
      size_type k = y.rows();
      if (k == 1)
        return true;
      for (size_type m = 0; m < k; ++m) {
        for (size_type n = m + 1; n < k; ++n) {
          if (fabs(y(m,n) - y(n,m)) > CONSTRAINT_TOLERANCE) {
            std::ostringstream message;
            message << name << " is not symmetric. " 
                    << name << "[" << m << "," << n << "] is %1%, but "
                    << name << "[" << n << "," << m 
                    << "] element is " << y(n,m);
            T_result tmp 
              = policies::raise_domain_error<T_y>(function,
                                                  message.str().c_str(),
                                                  y(m,n), Policy());
            if (result != 0)
              *result = tmp;
            return false;
          }
        }
      }
      return true;
    }


    template <typename T_y, typename T_result>
    inline bool check_symmetric(const char* function,
                const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                const char* name,
                T_result* result) {
      return check_symmetric(function,y,name,result,default_policy());
    }

    template <typename T>
    inline bool check_symmetric(const char* function,
                const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                const char* name,
                T* result = 0) {
      return check_symmetric(function,y,name,result,default_policy());
    }




    /**
     * Return <code>true</code> if the specified matrix is positive definite
     *
     * NOTE: symmetry is NOT checked by this function
     * 
     * @param function
     * @param y Matrix to test.
     * @param name
     * @param result
     * @return <code>true</code> if the matrix is positive definite.
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings (message has (0,0) item)
    template <typename T_y, typename T_result, class Policy>
    inline bool check_pos_definite(const char* function,
                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                  const char* name,
                  T_result* result,
                  const Policy&) {
      typedef 
        typename Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>::size_type 
        size_type;
      if (y.rows() == 1 && y(0,0) <= CONSTRAINT_TOLERANCE) {
        std::ostringstream message;
        message << name << " is not positive definite. " 
                << name << "(0,0) is %1%.";
        T_result tmp = policies::raise_domain_error<T_y>(function,
                                                         message.str().c_str(),
                                                         y(0,0), Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      Eigen::LDLT< Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic> > cholesky 
        = y.ldlt();
      if((cholesky.vectorD().array() <= CONSTRAINT_TOLERANCE).any())  {
        std::ostringstream message;
        message << name << " is not positive definite. " 
                << name << "(0,0) is %1%.";
        T_result tmp = policies::raise_domain_error<T_y>(function,
                                                         message.str().c_str(),
                                                         y(0,0), Policy());
        if (result != 0)
          *result = tmp;
        return false;
      }
      return true;
    }


    template <typename T_y, typename T_result>
    inline bool check_pos_definite(const char* function,
                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                  const char* name,
                  T_result* result) {
      return check_pos_definite(function,y,name,result,default_policy());
    }


    template <typename T>
    inline bool check_pos_definite(const char* function,
                  const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                  const char* name,
                  T* result = 0) {
      return check_pos_definite(function,y,name,result,default_policy());
    }




    /**
     * Return <code>true</code> if the specified matrix is a valid
     * covariance matrix.  A valid covariance matrix must be square,
     * symmetric, and positive definite.
     *
     * @param function
     * @param y Matrix to test.
     * @param name
     * @param result
     * @return <code>true</code> if the matrix is a valid covariance matrix.
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings
    template <typename T_y, typename T_result, class Policy>
    inline bool check_cov_matrix(const char* function,
                 const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const char* name,
                 T_result* result,
                 const Policy&) {
      if (!check_size_match(function, y.rows(), y.cols(), result, Policy())) 
        return false;
      if (!check_positive(function, y.rows(), "rows", result, Policy()))
        return false;
      if (!check_symmetric(function, y, "y", result, Policy()))
        return false;
      if (!check_pos_definite(function, y, "y", result, Policy()))
        return false;
      return true;
    }

    template <typename T_y, typename T_result>
    inline bool check_cov_matrix(const char* function,
                 const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const char* name,
                 T_result* result) {
      return check_cov_matrix(function,y,name,result,default_policy());
    }


    template <typename T>
    inline bool check_cov_matrix(const char* function,
                 const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                 const char* name,
                 T* result = 0) {
      return check_cov_matrix(function,y,name,result,default_policy());
    }



    /**
     * Return <code>true</code> if the specified matrix is a valid
     * correlation matrix.  A valid correlation matrix is symmetric,
     * has a unit diagonal (all 1 values), and has all values between
     * -1 and 1 (inclussive).  
     *
     * @param function 
     * @param y Matrix to test.
     * @param name 
     * @param result 
     * 
     * @return <code>true</code> if the specified matrix is a valid
     * correlation matrix.
     * @tparam T Type of scalar.
     */
    // FIXME: update warnings
    template <typename T_y, typename T_result, class Policy>
    inline bool check_corr_matrix(const char* function,
                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                  const char* name,
                  T_result* result,
                  const Policy&) {
      if (!check_size_match(function, y.rows(), y.cols(), result, Policy())) 
        return false;
      if (!check_positive(function, y.rows(), "rows", result, Policy()))
        return false;
      if (!check_symmetric(function, y, "y", result, Policy()))
        return false;
      for (typename Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>::size_type
             k = 0; k < y.rows(); ++k) {
        if (fabs(y(k,k) - 1.0) > CONSTRAINT_TOLERANCE) {
          std::ostringstream message;
          message << name << " is not a valid correlation matrix. " 
                  << name << "(" << k << "," << k 
                  << ") is %1%, but should be near 1.0";
          T_result tmp 
            = policies::raise_domain_error<T_y>(function,
                                                message.str().c_str(),
                                                y(k,k), Policy());
          if (result != 0)
            *result = tmp;
          return false;
        }
      }
      if (!check_pos_definite(function, y, "y", result, Policy()))
        return false;
      return true;
    }


    template <typename T_y, typename T_result>
    inline bool check_corr_matrix(const char* function,
                  const Eigen::Matrix<T_y,Eigen::Dynamic,Eigen::Dynamic>& y,
                  const char* name,
                  T_result* result) {
      return check_corr_matrix(function,y,name,result,default_policy());
    }

    template <typename T>
    inline bool check_corr_matrix(const char* function,
                  const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y,
                  const char* name,
                  T* result = 0) {
      return check_corr_matrix(function,y,name,result,default_policy());
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
      if (!check_size_match(function, Sigma.rows(), Sigma.cols(),
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
        std::string message(name);
        message += " is not a valid simplex.";
        message += " The sum of the elements is %1%, but should be 1.0";
        T_prob sum = theta.sum();
        T_result tmp = raise_domain_error<T_result,T_prob>(function, 
                                                           message.c_str(), 
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
