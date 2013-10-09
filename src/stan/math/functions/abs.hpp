#ifndef __STAN__MATH__FUNCTIONS__ABS_HPP__
#define __STAN__MATH__FUNCTIONS__ABS_HPP__

#include <cmath>

namespace stan {

  namespace math {

    /**
     * Return floating-point absolute value.
     *
     * Delegates to <code>fabs(double)</code> rather than
     * <code>std::abs(int)</code>.
     *
     * @param x scalar
     * @return absolute value of scalar
     */
    double abs(double x) {
      return std::fabs(x);
    }

  }
}

#endif
