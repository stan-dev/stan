#ifndef __STAN__MATH__VALIDATE_NON_NEGATIVE_REP_HPP__
#define __STAN__MATH__VALIDATE_NON_NEGATIVE_REP_HPP__

#include <stdexcept>
#include <sstream>
#include <string>

namespace stan {

  namespace math {

    /**
     * If argument is negative, throw a domain error indicating the
     * function with the specified name.
     *
     * @param n Integer to test for non-negativity.
     * @param fun Name of function in which test is being done.
     * @throw std::domain_error If integer is negative.
     */
    inline void validate_non_negative_rep(int n, const std::string& fun) {
      if (n >= 0) return;
      std::stringstream msg;
      msg << "error in " << fun
          << "; number of replications must be positive"
          << "; found n=" << n;
      throw std::domain_error(msg.str());
    }

  }
}

#endif
