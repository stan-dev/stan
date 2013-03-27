#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_UNIT_VECTOR_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_UNIT_VECTOR_HPP__

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/default_policy.hpp>
#include <stan/math/error_handling/raise_domain_error.hpp>
#include <stan/math/error_handling/matrix/constraint_tolerance.hpp>

namespace stan {
  namespace math {

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

  }
}
#endif
