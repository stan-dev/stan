#ifndef STAN__ERROR_HANDLING__SCALAR__DOM_ERR_HPP
#define STAN__ERROR_HANDLING__SCALAR__DOM_ERR_HPP

#include <typeinfo>
#ifdef BOOST_MSVC
#  pragma warning(push) // Quiet warnings in boost/format.hpp
#  pragma warning(disable: 4996) // _SCL_SECURE_NO_DEPRECATE
#  pragma warning(disable: 4512) // assignment operator could not be generated.
// And warnings in error handling:
#  pragma warning(disable: 4702) // unreachable code
// Note that this only occurs when the compiler can deduce code is unreachable,
// for example when policy macros are used to ignore errors rather than throw.
#endif
#include <boost/format.hpp>

#include <sstream>
#include <stdexcept>

namespace stan {
  namespace error_handling {

    namespace {
      // local output stream for pairs
      template <typename T1, typename T2>
      std::ostream& operator<<(std::ostream& o,
                               std::pair<T1,T2> xs) {
        o << '(' << xs.first << ", " << xs.second << ')';
        return o;
      }
    }


    // dom_err("function", "name", y, msg1, msg2);
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
    inline void dom_err(const char* function,
                        const char* name,
                        const T& y,
                        const char* msg1,
                        const char* msg2) {
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
    inline void dom_err(const char* function,
                        const char* name,
                        const T& y,
                        const char* msg1) {
      dom_err(function, name, y, msg1, "");
    }

  }
}
#endif
