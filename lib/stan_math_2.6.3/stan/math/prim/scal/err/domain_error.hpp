#ifndef STAN_MATH_PRIM_SCAL_ERR_DOMAIN_ERROR_HPP
#define STAN_MATH_PRIM_SCAL_ERR_DOMAIN_ERROR_HPP

#include <typeinfo>
#include <string>
#include <sstream>
#include <stdexcept>

namespace stan {
  namespace math {

    /**
     * Throw a domain error with a consistently formatted message.
     *
     * This is an abstraction for all Stan functions to use when throwing
     * domain errors. This will allow us to change the behavior for all
     * functions at once. (We've already changed behavior mulitple times up
     * to Stan v2.5.0.)
     *
     * The message is:
     * "<function>: <name> <msg1><y><msg2>"
     *
     * @tparam T Type of variable
     * @param function Name of the function
     * @param name Name of the variable
     * @param y Variable
     * @param msg1 Message to print before the variable
     * @param msg2 Message to print after the variable
     * @throw std::domain_error
     */
    template <typename T>
    inline void domain_error(const char* function,
                             const char* name,
                             const T& y,
                             const char* msg1,
                             const char* msg2) {
      std::ostringstream message;

      message << function << ": "
              << name << " "
              << msg1
              << y
              << msg2;

      throw std::domain_error(message.str());
    }

    /**
     * Throw a domain error with a consistently formatted message.
     *
     * This is an abstraction for all Stan functions to use when throwing
     * domain errors. This will allow us to change the behavior for all
     * functions at once. (We've already changed behavior mulitple times up
     * to Stan v2.5.0.)
     *
     * The message is:
     * "<function>: <name> <msg1><y>"
     *
     * @tparam T Type of variable
     * @param function Name of the function
     * @param name Name of the variable
     * @param y Variable
     * @param msg1 Message to print before the variable
     * @throw std::domain_error
     */
    template <typename T>
    inline void domain_error(const char* function,
                             const char* name,
                             const T& y,
                             const char* msg1) {
      domain_error(function, name, y, msg1, "");
    }

  }
}
#endif
