#ifndef __STAN__PROB__DISTRIBUTIONS_ERROR_HANDLING_HPP__
#define __STAN__PROB__DISTRIBUTIONS_ERROR_HANDLING_HPP__

#include <boost/math/policies/policy.hpp>

#include "stan/agrad/agrad.hpp"

namespace stan { 
  namespace prob {
    // reimplementing: #include <boost/math/distributions/detail/common_error_handling.hpp>
    
    template <typename T_x, typename T_result, class Policy>
    inline bool check_x(
			const char* function,
			const T_x& x,
			T_result* result,
			const Policy& pol) {
      if (!(boost::math::isfinite)(x)) {
	*result = boost::math::policies::raise_domain_error<T_x>(function,
								 "Random variate x is %1%, but must be finite!",
								 x, pol);
	return false;
      }
      return true;
      // Note that this test catches both infinity and NaN.
      // Some special cases permit x to be infinite, so these must be tested 1st,
      // leaving this test to catch any NaNs.  see Normal and cauchy for example.
    }
    
    template <typename T_result, class Policy>
    inline bool check_x(
			const char* function,
			const stan::agrad::var& x,
			T_result* result,
			const Policy& pol) {
      if(!(boost::math::isfinite)(x.val())) {
	*result = boost::math::policies::raise_domain_error<double>(function,
								    "Random variate x is %1%, but must be finite!", x.val(), pol);	  
	return false;
      }
      return true;
      // Note that this test catches both infinity and NaN.
      // Some special cases permit x to be infinite, so these must be tested 1st,
      // leaving this test to catch any NaNs.  see Normal and cauchy for example.
    } // bool check_x
    
    template <typename T_x, typename T_result, class Policy>
    inline bool check_x(
			const char* function,
			const std::vector<T_x>& x,
			T_result* result,
			const Policy& pol) {
      for (int i = 0; i < x.size(); i++) {
	if (!(boost::math::isfinite)(x[i])) {
	  *result = boost::math::policies::raise_domain_error<T_x>(function,
								   "Random variate x is %1%, but must be finite!",
								   x[i], pol);
	  return false;
	}
      }
      return true;
      // Note that this test catches both infinity and NaN.
      // Some special cases permit x to be infinite, so these must be tested 1st,
      // leaving this test to catch any NaNs.  see Normal and cauchy for example.
    }
    
    template <typename T_result, class Policy>
    inline bool check_x(
			const char* function,
			const std::vector<stan::agrad::var>& x,
			T_result* result,
			const Policy& pol) {
      for (int i = 0; i < x.size(); i++) {
	if (!(boost::math::isfinite)(x[i].val())) {
	  *result = boost::math::policies::raise_domain_error<double>(function,
								      "Random variate x is %1%, but must be finite!",
								      x[i].val(), pol);
	  return false;
	}
	return true;
      }
      return true;
      // Note that this test catches both infinity and NaN.
      // Some special cases permit x to be infinite, so these must be tested 1st,
      // leaving this test to catch any NaNs.  see Normal and cauchy for example.
    } // bool check_x


template <typename T_x, typename T_result, class Policy>
    inline bool check_x(
			const char* function,
			const Eigen::Matrix<T_x,Eigen::Dynamic,1>& x,
			T_result* result,
			const Policy& pol) {
      for (int i = 0; i < x.rows(); i++) {
	if (!(boost::math::isfinite)(x[i])) {
	  *result = boost::math::policies::raise_domain_error<T_x>(function,
								   "Random variate x is %1%, but must be finite!",
								   x[i], pol);
	  return false;
	}
      }
      return true;
      // Note that this test catches both infinity and NaN.
      // Some special cases permit x to be infinite, so these must be tested 1st,
      // leaving this test to catch any NaNs.  see Normal and cauchy for example.
    }
    
    template <typename T_result, class Policy>
    inline bool check_x(
			const char* function,
			const Eigen::Matrix<stan::agrad::var,Eigen::Dynamic,1>& x,
			T_result* result,
			const Policy& pol) {
      for (int i = 0; i < x.rows(); i++) {
	if (!(boost::math::isfinite)(x[i].val())) {
	  *result = boost::math::policies::raise_domain_error<double>(function,
								      "Random variate x is %1%, but must be finite!",
								      x[i].val(), pol);
	  return false;
	}
	return true;
      }
      return true;
      // Note that this test catches both infinity and NaN.
      // Some special cases permit x to be infinite, so these must be tested 1st,
      // leaving this test to catch any NaNs.  see Normal and cauchy for example.
    } // bool check_x


    template <typename T_scale, typename T_result, class Policy>
    inline bool check_scale(
			    const char* function,
			    const T_scale& scale,
			    T_result* result,
			    const Policy& pol) {
      if((scale <= 0) || !(boost::math::isfinite)(scale)) { // Assume scale == 0 is NOT valid for any distribution.
	*result = boost::math::policies::raise_domain_error<T_scale>(
								     function,
								     "Scale parameter is %1%, but must be > 0 !", scale, pol);
	return false;
      }
      return true;
    }

    template <typename T_result, class Policy>
    inline bool check_scale(
			    const char* function,
			    const stan::agrad::var& scale,
			    T_result* result,
			    const Policy& pol)
    {
      if((scale <= 0) || !(boost::math::isfinite)(scale.val())) { // Assume scale == 0 is NOT valid for any distribution.
	*result = boost::math::policies::raise_domain_error<double>(
								    function,
								    "Scale parameter is %1%, but must be > 0 !", scale.val(), pol);
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
      if(!(boost::math::isfinite)(location)) {
	*result = boost::math::policies::raise_domain_error<T_location>(
									function,
									"Location parameter is %1%, but must be finite!", location, pol);
	return false;
      }
      return true;
    }

    template <typename T_result, class Policy>
    inline bool check_location(
			       const char* function,
			       const stan::agrad::var& location,
			       T_result* result,
			       const Policy& pol) {
      if(!(boost::math::isfinite)(location.val()))
	{
	  *result = boost::math::policies::raise_domain_error<double>(
								      function,
								      "Location parameter is %1%, but must be finite!", location.val(), pol);
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
      if(!(boost::math::isfinite)(lb)) {
	*result = boost::math::policies::raise_domain_error<T_bound>(
									function,
									"Lower bound is %1%, but must be finite!", lb, pol);
	return false;
      }
      return true;
    }

    template <typename T_result, class Policy>
    inline bool check_lower_bound(
			       const char* function,
			       const stan::agrad::var& lb,
			       T_result* result,
			       const Policy& pol) {
      if(!(boost::math::isfinite)(lb.val()))
	{
	  *result = boost::math::policies::raise_domain_error<double>(
								      function,
								      "Lower bound is %1%, but must be finite!", lb.val(), pol);
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
      if(!(boost::math::isfinite)(ub)) {
	*result = boost::math::policies::raise_domain_error<T_bound>(
									function,
									"Upper bound is %1%, but must be finite!", ub, pol);
	return false;
      }
      return true;
    }

    template <typename T_result, class Policy>
    inline bool check_upper_bound(
			       const char* function,
			       const stan::agrad::var& ub,
			       T_result* result,
			       const Policy& pol) {
      if(!(boost::math::isfinite)(ub.val()))
	{
	  *result = boost::math::policies::raise_domain_error<double>(
								      function,
								      "Upper bound is %1%, but must be finite!", ub.val(), pol);
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
	*result = boost::math::policies::raise_domain_error<T_lb>(function,
								  "lower parameter is %1%, but must be less than upper!", lower, pol);
	return false;
      }
      return true;
    }


    template <typename T_ub, typename T_result, class Policy>
    inline bool check_bounds(
			     const char* function,
			     const stan::agrad::var& lower,
			     const T_ub& upper,
			     T_result* result,
			     const Policy& pol) {
      if (false == check_lower_bound(function, lower, result, pol))
	return false;
      if (false == check_upper_bound(function, upper, result, pol))
	return false;
      if (lower >= upper) {
	*result = boost::math::policies::raise_domain_error<double>(function,
								    "lower parameter is %1%, but must be less than upper!", lower.val(), pol);
	return false;
      }
      return true;
    }

    
    template <typename T_covar, typename T_result, class Policy>
    inline bool check_cov_matrix(
				 const char* function,
				 const Matrix<T_covar,Dynamic,Dynamic>& Sigma,
				 T_result* result,
				 const Policy& pol) {
      if (!stan::prob::cov_matrix_validate(Sigma)) {
	std::ostringstream stream;
	stream << "Sigma is not a valid covariance matrix. Sigma must be symmetric and positive semi-definite. Sigma: \n"<< Sigma << "\nSigma(0,0): %1%";
	*result = boost::math::policies::raise_domain_error<T_covar>(function,
								     stream.str().c_str(), 
								     Sigma(0,0),
								     pol);
	return false;
      }
      return true;
    }


    template <typename T_result, class Policy>
    inline bool check_cov_matrix(
				 const char* function,
				 const Matrix<stan::agrad::var,Dynamic,Dynamic>& Sigma,
				 T_result* result,
				 const Policy& pol) {
      if (!stan::prob::cov_matrix_validate(Sigma)) {
	std::ostringstream stream;
	stream << "Sigma is not a valid covariance matrix. Sigma must be symmetric and positive semi-definite. Sigma: \n"<< Sigma << "\nSigma(0,0): %1%";
	*result = boost::math::policies::raise_domain_error<double>(function,
								    stream.str().c_str(), 
								    Sigma(0,0),
								    pol);
	return false;
      }
      return true;
    }



  }
}
#endif
