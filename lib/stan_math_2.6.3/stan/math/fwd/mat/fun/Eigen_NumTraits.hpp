#ifndef STAN_MATH_FWD_MAT_FUN_EIGEN_NUMTRAITS_HPP
#define STAN_MATH_FWD_MAT_FUN_EIGEN_NUMTRAITS_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <limits>

namespace Eigen {

  /**
   * Numerical traits template override for Eigen for automatic
   * gradient variables.
   */
  template <typename T>
  struct NumTraits<stan::math::fvar<T> > {
    /**
     * Real-valued variables.
     *
     * Required for numerical traits.
     */
    typedef stan::math::fvar<T> Real;

    /**
     * Non-integer valued variables.
     *
     * Required for numerical traits.
     */
    typedef stan::math::fvar<T> NonInteger;

    /**
     * Nested variables.
     *
     * Required for numerical traits.
     */
    typedef stan::math::fvar<T> Nested;

    /**
     * Return standard library's epsilon for double-precision floating
     * point, <code>std::numeric_limits<double>::epsilon()</code>.
     *
     * @return Same epsilon as a <code>double</code>.
     */
    inline static Real epsilon() {
      return std::numeric_limits<double>::epsilon();
    }

    /**
     * Return dummy precision
     */
    inline static Real dummy_precision() {
      return 1e-12;  // copied from NumTraits.h values for double
    }

    /**
     * Return standard library's highest for double-precision floating
     * point, <code>std::numeric_limits&lt;double&gt;::max()</code>.
     *
     * @return Same highest value as a <code>double</code>.
     */
    inline static Real highest() {
      return std::numeric_limits<double>::max();
    }

    /**
     * Return standard library's lowest for double-precision floating
     * point, <code>&#45;std::numeric_limits&lt;double&gt;::max()</code>.
     *
     * @return Same lowest value as a <code>double</code>.
     */
    inline static Real lowest() {
      return -std::numeric_limits<double>::max();
    }

    /**
     * Properties for automatic differentiation variables
     * read by Eigen matrix library.
     */
    enum {
      IsInteger = 0,
      IsSigned = 1,
      IsComplex = 0,
      RequireInitialization = 1,
      ReadCost = 1,
      AddCost = 1,
      MulCost = 1,
      HasFloatingPoint = 1
    };
  };

  namespace internal {
    /**
     * Implemented this for printing to stream.
     */
    template<typename T>
    struct significant_decimals_default_impl<stan::math::fvar<T>, false> {
      static inline int run() {
        using std::ceil;
        using std::log;
        return cast<double, int>
          (ceil(-log(std::numeric_limits<double>::epsilon())
                / log(10.0)));
      }
    };

  }
}

#endif
