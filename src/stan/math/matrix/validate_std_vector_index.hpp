#ifndef __STAN__MATH__MATRIX__VALIDATE_STD_VECTOR_INDEX_HPP__
#define __STAN__MATH__MATRIX__VALIDATE_STD_VECTOR_INDEX_HPP__

#include <sstream>
#include <stdexcept>
#include <vector>

namespace stan {
  namespace math {
    
    template <typename T>
    void validate_std_vector_index(const std::vector<T>& sv,
                                    size_t j,
                                    const char* msg) {
      if (j > 0 && j <=  sv.size()) return;
      std::stringstream ss;
      ss << "require 0 < index <= vector size" << msg;
      ss << "; found vector size=" << sv.size()
         << "; index j=" << j;
      throw std::domain_error(ss.str());
    }
    
  }
}

#endif
