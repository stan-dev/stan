#ifndef STAN__MATH__FUNCTIONS__MAX_HPP
#define STAN__MATH__FUNCTIONS__MAX_HPP

namespace stan {
  namespace math {

    inline double max(const double a, const double b) { 
      return a > b ? a : b; 
    }

  }
}

#endif
