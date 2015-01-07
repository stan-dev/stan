#ifndef STAN__ERROR_HANDLING__MATRIX__VALIDATE_NON_NEGATIVE_INDEX_HPP
#define STAN__ERROR_HANDLING__MATRIX__VALIDATE_NON_NEGATIVE_INDEX_HPP

#include <sstream>
#include <stdexcept>

namespace stan {
  namespace error_handling {

    inline 
    void 
    validate_non_negative_index(const std::string& var_name,
                                const std::string& expr,
                                int val) {
      if (val < 0) {
        std::stringstream msg;
        msg << "Found negative dimension size in variable declaration"
            << "; variable=" << var_name
            << "; dimension size expression=" << expr
            << "; expression value=" << val;
        throw std::invalid_argument(msg.str());
      }
    }

  }
}
#endif
