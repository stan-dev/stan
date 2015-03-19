#ifndef STAN__MATH__PRIM__SCAL__FUN__LUB_FREE_HPP
#define STAN__MATH__PRIM__SCAL__FUN__LUB_FREE_HPP

#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/fun/logit.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>

namespace stan {

  namespace prob {

    /**
     * Return the unconstrained scalar that transforms to the
     * specified lower- and upper-bounded scalar given the specified
     * bounds.
     *
     * <p>The transfrom in <code>lub_constrain(T,double,double)</code>,
     * is reversed by a transformed and scaled logit,
     *
     * <p>\f$f^{-1}(y) = \mbox{logit}(\frac{y - L}{U - L})\f$
     *
     * where \f$U\f$ and \f$L\f$ are the lower and upper bounds.
     *
     * <p>If the lower bound is negative infinity and upper bound finite,
     * this function reduces to <code>ub_free(y,ub)</code>.  If
     * the upper bound is positive infinity and the lower bound
     * finite, this function reduces to
     * <code>lb_free(x,lb)</code>.  If the upper bound is
     * positive infinity and the lower bound negative infinity,
     * this function reduces to <code>identity_free(y)</code>.
     *
     * @tparam T Type of scalar.
     * @param y Scalar input.
     * @param lb Lower bound.
     * @param ub Upper bound.
     * @return The free scalar that transforms to the input scalar
     * given the bounds.
     * @throw std::invalid_argument if the lower bound is greater than
     *   the upper bound, y is less than the lower bound, or y is
     *   greater than the upper bound
     */
    template <typename T, typename TL, typename TU>
    inline
    typename boost::math::tools::promote_args<T,TL,TU>::type
    lub_free(const T y, TL lb, TU ub) {
      using stan::math::logit;
      stan::math::check_bounded<T, TL, TU>
        ("stan::prob::lub_free",
         "Bounded variable",
         y, lb, ub);
      if (lb == -std::numeric_limits<double>::infinity())
        return ub_free(y,ub);
      if (ub == std::numeric_limits<double>::infinity())
        return lb_free(y,lb);
      return logit((y - lb) / (ub - lb));
    }

  }

}

#endif
