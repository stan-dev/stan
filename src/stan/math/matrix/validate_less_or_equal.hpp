#ifndef __STAN__MATH__MATRIX__VALIDATE_LESS_OR_EQUAL_HPP__
#define __STAN__MATH__MATRIX__VALIDATE_LESS_OR_EQUAL_HPP__

#include <sstream>
#include <stdexcept>

namespace stan {
  namespace math {
    
    template <typename T1, typename T2>
    inline
    void validate_less_or_equal(const T1& x, const T2& y,
                                const char* x_name, const char* y_name, 
                                const char* fun_name) {
      if (x <= y) return;
      std::stringstream ss;
      ss << "require " << x_name << " <= " << y_name
         << " in " << fun_name
         << "; found " << x_name << "=" << x
         << ", " << y_name << "=" << y;
      throw std::domain_error(ss.str());
    }
    
  }
}
#endif
