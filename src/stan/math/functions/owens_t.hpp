#ifndef STAN__MATH__FUNCTIONS__OWENS__T_HPP
#define STAN__MATH__FUNCTIONS__OWENS__T_HPP

#include <boost/math/tools/promotion.hpp>
#include <boost/math/special_functions/owens_t.hpp>

namespace stan {
  namespace math {

 
    /** 
     * The Owen's T function of h and a.
     *
     * Used to compute the cumulative density function for the skew normal
     * distribution.
     * 
     * @tparam T1 Type of first argument.
     * @tparam T2 Type of second argument.
     * @param h First argument
     * @param a Second argument
     * @return The Owen's T function.
     */
    template <typename T1, typename T2>
    inline typename boost::math::tools::promote_args<T1,T2>::type
    owens_t(const T1& h, const T2& a) {
      return boost::math::owens_t(h, a);
    }
  }
}

#endif
