#ifndef STAN_MATH_PRIM_SCAL_FUN_LB_FREE_HPP
#define STAN_MATH_PRIM_SCAL_FUN_LB_FREE_HPP

#include <stan/math/prim/scal/fun/identity_free.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <boost/math/tools/promotion.hpp>
#include <cmath>
#include <limits>

namespace stan {

  namespace math {

    /**
     * Return the unconstrained value that produces the specified
     * lower-bound constrained value.
     *
     * If the lower bound is negative infinity, it is ignored and
     * the function reduces to <code>identity_free(y)</code>.
     *
     * @param y Input scalar.
     * @param lb Lower bound.
     * @return Unconstrained value that produces the input when
     * constrained.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     * @throw std::domain_error if y is lower than the lower bound.
     */
    template <typename T, typename TL>
    inline
    typename boost::math::tools::promote_args<T, TL>::type
    lb_free(const T y, const TL lb) {
      using std::log;
      if (lb == -std::numeric_limits<double>::infinity())
        return identity_free(y);
      stan::math::check_greater_or_equal("stan::math::lb_free",
                                         "Lower bounded variable", y, lb);
      return log(y - lb);
    }

  }

}

#endif
