#ifndef STAN__ERROR_HANDLING__MATRIX__CONSTRAINT_TOLERANCE_HPP
#define STAN__ERROR_HANDLING__MATRIX__CONSTRAINT_TOLERANCE_HPP

namespace stan {
  namespace error_handling {

    /**
     * The tolerance for checking arithmetic bounds In rank and in
     * simplexes.  The default value is <code>1E-8</code>.
     */
    const double CONSTRAINT_TOLERANCE = 1E-8;

  }
}
#endif
