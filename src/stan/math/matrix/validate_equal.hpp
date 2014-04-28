#ifndef __STAN__MATH__MATRIX__VALIDATE_EQUAL_HPP__
#define __STAN__MATH__MATRIX__VALIDATE_EQUAL_HPP__

#include <sstream>
#include <stdexcept>

namespace stan {
  namespace math {

    template <typename T1, typename T2>
    inline
    void validate_equal(const T1& x, const T2& y,
                                   const char* x_name, const char* y_name, 
                                   const char* fun_name) {
      if (x == y) return;
      std::stringstream ss;
      ss << "error in call to " << fun_name
         << "; require " << x_name << " equal to " << y_name
         << "; found " << x_name << "=" << x
         << ", " << y_name << "=" << y;
      throw std::domain_error(ss.str());
    }    
    
  }
}
#endif
