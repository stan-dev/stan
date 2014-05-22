#ifndef __STAN__MATH__ERROR_HANDLING__MATRIX__CONSTRAINT_TOLERANCE_HPP__
#define __STAN__MATH__ERROR_HANDLING__MATRIX__CONSTRAINT_TOLERANCE_HPP__

namespace stan {
  namespace math {

    /**
     * The tolerance for checking arithmetic bounds In rank and in
     * simplexes.  The default value is <code>1E-8</code>.
     */
    const double CONSTRAINT_TOLERANCE = 1E-8;

  }
}
#endif
