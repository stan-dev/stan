#ifndef __STAN__PROB__TRAITS_HPP__
#define __STAN__PROB__TRAITS_HPP__

#include <stan/meta/traits.hpp>

namespace stan {

  namespace prob {

    /**
     * Template metaprogram to calculate whether a summand
     * needs to be included in a proportional (log) probability 
     * calculations.  For usage, the first boolean parameter
     * should be set to <code>true</code> if calculating
     * a term up to proportionality.  Other type parameters
     * should be included for all of the types of variables
     * in a term.
     *
     * The <code>value</code> enum will be <code>true</code> if the
     * <code>propto</code> parameter is <code>false</code> or if any
     * of the other template arguments are not constants as defined by
     * <code>stan::is_constant<T></code>.  

     * @tparam propto <code>true</code> if calculating up to a
     * proportionality constant.
     * @tparam T1 First 
     */
    template <bool propto,
	      typename T1=double, typename T2=double, 
	      typename T3=double, typename T4=double,
	      typename T5=double, typename T6=double>
    struct include_summand {
      /**
       * <code>true</code> if a term with the specified propto
       * value and subterm types should be included in a proportionality
       * calculation.
       */
      enum { 
	value =  ( !propto
		   || !stan::is_constant<T1>::value
		   || !stan::is_constant<T2>::value
		   || !stan::is_constant<T3>::value
		   || !stan::is_constant<T4>::value 
		   || !stan::is_constant<T5>::value 
		   || !stan::is_constant<T6>::value  )
      };
    };

  }
}

#endif
