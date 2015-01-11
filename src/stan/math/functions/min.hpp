#ifndef STAN__MATH__FUNCTIONS__MIN_HPP
#define STAN__MATH__FUNCTIONS__MIN_HPP

namespace stan {
  namespace math {

    inline double min(const double a, const double b) { 
      return a < b ? a : b; 
    }

  }
}

#endif
