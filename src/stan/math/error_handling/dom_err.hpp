#ifndef STAN__MATH__ERROR_HANDLING__DOM_ERR_HPP
#define STAN__MATH__ERROR_HANDLING__DOM_ERR_HPP

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
  namespace math {

    namespace {
      // local output stream for pairs
      template <typename T1, typename T2>
      std::ostream& operator<<(std::ostream& o,
                               std::pair<T1,T2> xs) {
        o << '(' << xs.first << ", " << xs.second << ')';
        return o;
      }
    }

    // currently ignoring T_result
    template <typename T,
              typename T_result,
              typename T_msg>
    inline bool dom_err(const char* function,
                        const T& y,
                        const char* name,
                        const char* error_msg,
                        const T_msg error_msg2,
                        T_result* result) {
      std::ostringstream msg_o;
      msg_o << name << error_msg << error_msg2;
      
      std::string msg;
      msg += (boost::format(function) % typeid(T).name()).str();
      msg += ": ";
      msg += msg_o.str();
      
      throw std::domain_error((boost::format(msg) % y).str());

      return false;
    }
    

  }
}
#endif
