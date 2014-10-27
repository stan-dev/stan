#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_SIMPLEX_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_SIMPLEX_HPP

#include <sstream>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/error_handling/matrix/constraint_tolerance.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/meta/index_type.hpp>
#include <stan/meta/traits.hpp>


namespace stan {

  namespace error_handling {

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
     * @return throws if any element is nan.
     */
    template <typename T_prob, typename T_result>
    bool check_simplex(const char* function,
                       const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta,
                       const char* name,
                       T_result* result) {
      using Eigen::Dynamic;
      using Eigen::Matrix;
      typedef typename index_type<Matrix<T_prob,Dynamic,1> >::type size_t;

      if (theta.size() == 0) {
        std::stringstream msg;
        msg << " is not a valid simplex. " 
            << "length(" << name << ") = %1%";
        std::string tmp(msg.str());
        return dom_err(function,0,name,
                       tmp.c_str(),"",
                       result);
      }
      if (!(fabs(1.0 - theta.sum()) <= CONSTRAINT_TOLERANCE)) {
        std::stringstream msg;
        T_prob sum = theta.sum();
        msg << " is not a valid simplex.";
        msg.precision(10);
        msg << " sum(" << name << ") = " << sum
            << ", but should be %1%";
        std::string tmp(msg.str());
        return dom_err(function,1.0,name,
                       tmp.c_str(),"",
                       result);
      }
      for (size_t n = 0; n < theta.size(); n++) {
        if (!(theta[n] >= 0)) {
          std::ostringstream stream;
          stream << " is not a valid simplex. "
                 << name << "[" << n + stan::error_index::value << "]"
                 << " = %1%, but should be greater than or equal to 0";
          std::string tmp(stream.str());
          return dom_err(function,theta[n],name,
                         tmp.c_str(),"",
                         result);
        }
      }
      return true;
    }                         

  }
}
#endif
