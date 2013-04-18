#ifndef __STAN__MATH__MATRIX__VALIDATE_NONZERO_SIZE_HPP__
#define __STAN__MATH__MATRIX__VALIDATE_NONZERO_SIZE_HPP__

#include <sstream>
#include <stdexcept>

namespace stan {
  namespace math {
    
    template <typename T>
    inline void validate_nonzero_size(const T& x, const char* msg) {
      if (x.size() > 0) return;
      std::stringstream ss;
      ss << "require non-zero size for " << msg
         << "found size=" << x.size();
      throw std::domain_error(ss.str());
    }
    
  }
}
#endif
