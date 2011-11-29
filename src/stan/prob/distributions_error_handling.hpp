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
      
  }}

#endif
