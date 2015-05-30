#ifndef STAN_MATH_PRIM_SCAL_META_INCLUDE_SUMMAND_HPP
#define STAN_MATH_PRIM_SCAL_META_INCLUDE_SUMMAND_HPP

#include <stan/math/prim/scal/meta/is_constant.hpp>
#include <stan/math/prim/scal/meta/scalar_type.hpp>
#include <boost/math/tools/promotion.hpp>

namespace stan {

  namespace math {

    /**
     * Template metaprogram to calculate whether a summand
     * needs to be included in a proportional (log) probability
     * calculation.  For usage, the first boolean parameter
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
              typename T1 = double, typename T2 = double,
              typename T3 = double, typename T4 = double,
              typename T5 = double, typename T6 = double,
              typename T7 = double, typename T8 = double,
              typename T9 = double, typename T10 = double>
    struct include_summand {
      /**
       * <code>true</code> if a term with the specified propto
       * value and subterm types should be included in a proportionality
       * calculation.
       */
      enum {
        value =  (!propto
                  || !stan::is_constant<typename scalar_type<T1>::type>::value
                  || !stan::is_constant<typename scalar_type<T2>::type>::value
                  || !stan::is_constant<typename scalar_type<T3>::type>::value
                  || !stan::is_constant<typename scalar_type<T4>::type>::value
                  || !stan::is_constant<typename scalar_type<T5>::type>::value
                  || !stan::is_constant<typename scalar_type<T6>::type>::value
                  || !stan::is_constant<typename scalar_type<T7>::type>::value
                  || !stan::is_constant<typename scalar_type<T8>::type>::value
                  || !stan::is_constant<typename scalar_type<T9>::type>::value
                  || !stan::is_constant<typename scalar_type<T10>::type>::value
                  )
      };
    };


  }

}

#endif
