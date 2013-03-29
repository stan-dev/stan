#ifndef __STAN__MATH__FUNCTIONS__MIN_HPP__
#define __STAN__MATH__FUNCTIONS__MIN_HPP__

namespace stan {
  namespace math {

    inline double min(const double a, const double b) { 
      return a < b ? a : b; 
    }

  }
}

#endif
