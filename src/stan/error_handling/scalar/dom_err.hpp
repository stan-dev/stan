#ifndef STAN__ERROR_HANDLING__SCALAR__DOM_ERR_HPP
#define STAN__ERROR_HANDLING__SCALAR__DOM_ERR_HPP

#include <typeinfo>
#include <string>
#include <sstream>
#include <stdexcept>

namespace stan {
  namespace error_handling {

    /**
     * Throw a domain error with a consistently formatted message.
     * 
     * This is an abstraction for all Stan functions to use when throwing
     * domain errors. This will allow us to change the behavior for all
     * functions at once. (We've already changed behavior mulitple times up
     * to Stan v2.5.0.)
     *
     * The message is:
     * "<function>(<typeid(T)>.name()>): <name> <msg1><y><msg2>"
     *
     * @tparam T Type of variable
     * @param function Name of the function
     * @param name Name of the variable
     * @param y Variable
     * @param msg1 Message to print before the variable
     * @param msg2 Message to print after the variable
     */
    template <typename T>
    inline void dom_err(const std::string& function,
                        const std::string& name,
                        const T& y,
                        const std::string& msg1,
                        const std::string& msg2) {
      std::ostringstream message;
      
      message << function << "(" << typeid(T).name() << "): "
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
     * "<function>(<typeid(T)>.name()>): <name> <msg1><y>"
     *
     * @tparam T Type of variable
     * @param function Name of the function
     * @param name Name of the variable
     * @param y Variable
     * @param msg1 Message to print before the variable
     */
    template <typename T>
    inline void dom_err(const std::string& function,
                        const std::string& name,
                        const T& y,
                        const std::string& msg1) {
      dom_err(function, name, y, msg1, "");
    }

  }
}
#endif
