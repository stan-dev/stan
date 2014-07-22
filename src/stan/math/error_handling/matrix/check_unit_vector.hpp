#ifndef STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_UNIT_VECTOR_HPP
#define STAN__MATH__ERROR_HANDLING__MATRIX__CHECK_UNIT_VECTOR_HPP

#include <sstream>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/dom_err.hpp>
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
    template <typename T_prob, typename T_result>
    bool check_unit_vector(const char* function,
                           const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta,
                           const char* name,
                           T_result* result) {
      typedef typename Eigen::Matrix<T_prob,Eigen::Dynamic,1>::size_type size_t;
      if (theta.size() == 0) {
        std::string message(name);
        message += " is not a valid unit vector. %1% elements in the vector.";
        return dom_err(function,0,name,
                       message.c_str(),"",
                       result);
      }
      T_prob ssq = theta.squaredNorm();
      if (fabs(1.0 - ssq) > CONSTRAINT_TOLERANCE) {
        std::stringstream msg;
        msg << "in function check_unit_vector(%1%), ";
        msg << name << " is not a valid unit vector.";
        msg << " The sum of the squares of the elements should be 1, but is " 
            << ssq;
        std::string tmp(msg.str());
        return dom_err(function,ssq,name,
                       tmp.c_str(),"",
                       result);
      }
      return true;
    }


    template <typename T>
    inline bool check_unit_vector(const char* function,
                                  const Eigen::Matrix<T,Eigen::Dynamic,1>& theta,
                                  const char* name,
                                  T* result = 0) {
      return check_unit_vector<T,T>(function,theta,name,result);
    }


  }
}
#endif
