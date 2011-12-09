#ifndef __STAN__PROB__DISTRIBUTIONS_ERROR_HANDLING_HPP__
#define __STAN__PROB__DISTRIBUTIONS_ERROR_HANDLING_HPP__

#include <limits>
#include <boost/math/policies/policy.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/math/distributions/detail/common_error_handling.hpp>
#include <Eigen/Dense>
#include "stan/agrad/agrad.hpp"
#include "stan/prob/transform.hpp"

namespace stan { 
  namespace prob {
    // reimplementing a lot from: #include <boost/math/distributions/detail/common_error_handling.hpp>
    template <typename T>
    inline double convert (const T& x) {
      return x;
    }
    template <>
    inline double convert (const stan::agrad::var& x) {
      return x.val();
    }


    template <typename T_x, typename T_result, class Policy>
    inline bool check_x(
			const char* function,
			const T_x& x,
			T_result* result,
			const Policy& pol) {
      if (!(boost::math::isfinite)(convert(x))) {
	*result = boost::math::policies::raise_domain_error<double>(function,
								 "Random variate x is %1%, but must be finite!",
								 convert(x), pol);
	return false;
      }
      return true;
      // Note that this test catches both infinity and NaN.
      // Some special cases permit x to be infinite, so these must be tested 1st,
      // leaving this test to catch any NaNs.  see Normal and cauchy for example.
    }
        
    template <typename T_x, typename T_result, class Policy>
    inline bool check_x(
			const char* function,
			const std::vector<T_x>& x,
			T_result* result,
			const Policy& pol) {
      for (int i = 0; i < x.size(); i++) {
	if (!(boost::math::isfinite)(convert(x[i]))) {
	  *result = boost::math::policies::raise_domain_error<double>(function,
								   "Random variate x is %1%, but must be finite!",
								   convert(x[i]), pol);
	  return false;
	}
      }
      return true;
      // Note that this test catches both infinity and NaN.
      // Some special cases permit x to be infinite, so these must be tested 1st,
      // leaving this test to catch any NaNs.  see Normal and cauchy for example.
    }
    
    template <typename T_x, typename T_result, class Policy>
    inline bool check_x(
			const char* function,
			const Eigen::Matrix<T_x,Eigen::Dynamic,1>& x,
			T_result* result,
			const Policy& pol) {
      for (int i = 0; i < x.rows(); i++) {
	if (!(boost::math::isfinite)(convert(x[i]))) {
	  *result = boost::math::policies::raise_domain_error<double>(function,
								   "Random variate x is %1%, but must be finite!",
								   convert(x[i]), pol);
	  return false;
	}
      }
      return true;
      // Note that this test catches both infinity and NaN.
      // Some special cases permit x to be infinite, so these must be tested 1st,
      // leaving this test to catch any NaNs.  see Normal and cauchy for example.
    }

    template <typename T_x, typename T_result, class Policy>
    inline bool check_bounded_x(
			const char* function,
			const T_x& x,
			const double low,
			const double high,
			T_result* result,
			const Policy& pol) {
      if (!(boost::math::isfinite)(convert(x)) || !(low <= x && x <= high)) {
	std::string msg ("Random variate x is %1%, but must be finite and between ");
	msg += low;
	msg += " and ";
	msg += high;
	*result = boost::math::policies::raise_domain_error<double>(function,
								    msg.c_str(),
								    convert(x), pol);
	return false;
      }
      return true;
      // Note that this test catches both infinity and NaN.
      // Some special cases permit x to be infinite, so these must be tested 1st,
      // leaving this test to catch any NaNs.  see Normal and cauchy for example.
    }
    

    template <typename T_scale, typename T_result, class Policy>
    inline bool check_scale(
			    const char* function,
			    const T_scale& scale,
			    T_result* result,
			    const Policy& pol) {
      if(!(scale > 0) || !(boost::math::isfinite)(convert(scale))) { // Assume scale == 0 is NOT valid for any distribution.
	*result = boost::math::policies::raise_domain_error<double>(
								     function,
								     "Scale parameter is %1%, but must be > 0 !", convert(scale), pol);
	return false;
      }
      return true;
    }

    template <typename T_inv_scale, typename T_result, class Policy>
    inline bool check_inv_scale(
			    const char* function,
			    const T_inv_scale& invScale,
			    T_result* result,
			    const Policy& pol) {
      if(!(invScale > 0) || !(boost::math::isfinite)(convert(invScale))) { // Assume scale == 0 is NOT valid for any distribution.
	*result = boost::math::policies::raise_domain_error<double>(
								     function,
								     "Inverse scale parameter is %1%, but must be > 0 !", convert(invScale), pol);
	return false;
      }
      return true;
    }


    template <typename T_x, typename T_result, class Policy>
    inline bool check_nonnegative(
				  const char* function,
				  const T_x& x,
				  const char* name,
				  T_result* result,
				  const Policy& pol) {
      if(!(boost::math::isfinite)(convert(x)) || !(x >= 0)) {
	std::string message(name);
	message += " is %1%, but must be finite and >= 0!";
	*result = boost::math::policies::raise_domain_error<double>(
								 function,
								 message.c_str(), convert(x), pol);
	return false;
      }
      return true;
    }

    template <typename T_result, class Policy>
    inline bool check_nonnegative(
				  const char* function,
				  const unsigned int& x,
				  const char* name,
				  T_result* result,
				  const Policy& pol) {
      if (std::numeric_limits< unsigned int >::has_infinity &&
	  x == std::numeric_limits< unsigned int>::infinity() ) {
	std::string message(name);
	message += " is %1%, but must be finite and >= 0!";
	*result = boost::math::policies::raise_domain_error<double>(
								 function,
								 message.c_str(), convert(x), pol);
	return false;
      }
      return true;
    }


    template <typename T_x, typename T_result, class Policy>
    inline bool check_positive(
				  const char* function,
				  const T_x& x,
				  const char* name,
				  T_result* result,
				  const Policy& pol) {
      if(!(boost::math::isfinite)(convert(x)) || !(x > 0)) {
	std::string message(name);
	message += " is %1%, but must be finite and > 0!";
	*result = boost::math::policies::raise_domain_error<double>(
								 function,
								 message.c_str(), convert(x), pol);
	return false;
      }
      return true;
    }

    template <typename T_location, typename T_result, class Policy>
    inline bool check_location(
			       const char* function,
			       const T_location& location,
			       T_result* result,
			       const Policy& pol) {
      if(!(boost::math::isfinite)(convert(location))) {
	*result = boost::math::policies::raise_domain_error<double>(
									function,
									"Location parameter is %1%, but must be finite!", convert(location), pol);
	return false;
      }
      return true;
    }

    template <typename T_bound, typename T_result, class Policy>
    inline bool check_lower_bound(
				  const char* function,
				  const T_bound& lb,
				  T_result* result,
				  const Policy& pol) {
      if(!(boost::math::isfinite)(convert(lb))) {
	*result = boost::math::policies::raise_domain_error<double>(
								     function,
								     "Lower bound is %1%, but must be finite!", convert(lb), pol);
	return false;
      }
      return true;
    }


    template <typename T_bound, typename T_result, class Policy>
    inline bool check_upper_bound(
				  const char* function,
				  const T_bound& ub,
				  T_result* result,
				  const Policy& pol) {
      if(!(boost::math::isfinite)(convert(ub))) {
	*result = boost::math::policies::raise_domain_error<double>(
								     function,
								     "Upper bound is %1%, but must be finite!", convert(ub), pol);
	return false;
      }
      return true;
    }

    template <typename T_lb, typename T_ub, typename T_result, class Policy>
    inline bool check_bounds(
			     const char* function,
			     const T_lb& lower,
			     const T_ub& upper,
			     T_result* result,
			     const Policy& pol) {
      if (false == check_lower_bound(function, lower, result, pol))
	return false;
      if (false == check_upper_bound(function, upper, result, pol))
	return false;
      if (lower >= upper) {
	*result = boost::math::policies::raise_domain_error<double>(function,
								  "lower parameter is %1%, but must be less than upper!", convert(lower), pol);
	return false;
      }
      return true;
    }

    
    template <typename T_covar, typename T_result, class Policy>
    inline bool check_cov_matrix(
				 const char* function,
				 const Eigen::Matrix<T_covar,Eigen::Dynamic,Eigen::Dynamic>& Sigma,
				 T_result* result,
				 const Policy& pol) {
      if (!stan::prob::cov_matrix_validate(Sigma)) {
	std::string message ("Sigma is not a valid covariance matrix. Sigma must be symmetric and positive semi-definite. Sigma: \n");
	message += Sigma;
	message += "\nSigma(0,0): %1%";
	*result = boost::math::policies::raise_domain_error<double>(function,
								     message.c_str(), 
								     convert(Sigma(0,0)),
								     pol);
	return false;
      }
      return true;
    }


  }
}
#endif
