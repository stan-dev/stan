#ifndef STAN_MATH_PRIM_SCAL_FUN_LB_CONSTRAIN_HPP
#define STAN_MATH_PRIM_SCAL_FUN_LB_CONSTRAIN_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/scal/fun/identity_constrain.hpp>
#include <cmath>
#include <limits>

namespace stan {

  namespace math {
    // LOWER BOUND

    /**
     * Return the lower-bounded value for the specified unconstrained input
     * and specified lower bound.
     *
     * <p>The transform applied is
     *
     * <p>\f$f(x) = \exp(x) + L\f$
     *
     * <p>where \f$L\f$ is the constant lower bound.
     *
     * <p>If the lower bound is negative infinity, this function
     * reduces to <code>identity_constrain(x)</code>.
     *
     * @param x Unconstrained scalar input.
     * @param lb Lower-bound on constrained ouptut.
     * @return Lower-bound constrained value correspdonding to inputs.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     */
    template <typename T, typename TL>
    inline
    T lb_constrain(const T x, const TL lb) {
      using std::exp;
      if (lb == -std::numeric_limits<double>::infinity())
        return identity_constrain(x);
      return exp(x) + lb;
    }

    /**
     * Return the lower-bounded value for the speicifed unconstrained
     * input and specified lower bound, incrementing the specified
     * reference with the log absolute Jacobian determinant of the
     * transform.
     *
     * If the lower bound is negative infinity, this function
     * reduces to <code>identity_constraint(x, lp)</code>.
     *
     * @param x Unconstrained scalar input.
     * @param lb Lower-bound on output.
     * @param lp Reference to log probability to increment.
     * @return Loer-bound constrained value corresponding to inputs.
     * @tparam T Type of scalar.
     * @tparam TL Type of lower bound.
     */
    template <typename T, typename TL>
    inline
    typename boost::math::tools::promote_args<T, TL>::type
    lb_constrain(const T x, const TL lb, T& lp) {
      using std::exp;
      if (lb == -std::numeric_limits<double>::infinity())
        return identity_constrain(x, lp);
      lp += x;
      return exp(x) + lb;
    }


  }

}

#endif
