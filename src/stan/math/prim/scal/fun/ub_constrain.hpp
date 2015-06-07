#ifndef STAN_MATH_PRIM_SCAL_FUN_UB_CONSTRAIN_HPP
#define STAN_MATH_PRIM_SCAL_FUN_UB_CONSTRAIN_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/scal/fun/identity_constrain.hpp>
#include <cmath>
#include <limits>

namespace stan {

  namespace math {

    // UPPER BOUND

    /**
     * Return the upper-bounded value for the specified unconstrained
     * scalar and upper bound.
     *
     * <p>The transform is
     *
     * <p>\f$f(x) = U - \exp(x)\f$
     *
     * <p>where \f$U\f$ is the upper bound.
     *
     * If the upper bound is positive infinity, this function
     * reduces to <code>identity_constrain(x)</code>.
     *
     * @param x Free scalar.
     * @param ub Upper bound.
     * @return Transformed scalar with specified upper bound.
     * @tparam T Type of scalar.
     * @tparam TU Type of upper bound.
     */
    template <typename T, typename TU>
    inline
    typename boost::math::tools::promote_args<T, TU>::type
    ub_constrain(const T x, const TU ub) {
      using std::exp;
      if (ub == std::numeric_limits<double>::infinity())
        return identity_constrain(x);
      return ub - exp(x);
    }

    /**
     * Return the upper-bounded value for the specified unconstrained
     * scalar and upper bound and increment the specified log
     * probability reference with the log absolute Jacobian
     * determinant of the transform.
     *
     * <p>The transform is as specified for
     * <code>ub_constrain(T, double)</code>.  The log absolute Jacobian
     * determinant is
     *
     * <p>\f$ \log | \frac{d}{dx} -\mbox{exp}(x) + U |
     *     = \log | -\mbox{exp}(x) + 0 | = x\f$.
     *
     * If the upper bound is positive infinity, this function
     * reduces to <code>identity_constrain(x, lp)</code>.
     *
     * @param x Free scalar.
     * @param ub Upper bound.
     * @param lp Log probability reference.
     * @return Transformed scalar with specified upper bound.
     * @tparam T Type of scalar.
     * @tparam TU Type of upper bound.
     */
    template <typename T, typename TU>
    inline
    typename boost::math::tools::promote_args<T, TU>::type
    ub_constrain(const T x, const TU ub, T& lp) {
      using std::exp;
      if (ub == std::numeric_limits<double>::infinity())
        return identity_constrain(x, lp);
      lp += x;
      return ub - exp(x);
    }

  }

}

#endif
