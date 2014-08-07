#ifndef STAN__MATH__ERROR_HANDLING__DOM_ERR_VEC_HPP
#define STAN__MATH__ERROR_HANDLING__DOM_ERR_VEC_HPP

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

#include <stan/meta/traits.hpp>
#include <sstream>
#include <stdexcept>
#include <stan/math/error_handling/dom_err.hpp>

namespace stan {
  namespace math {


    // currently ignoring T_result
    template <typename T,
              typename T_result,
              typename T_msg>
    inline bool dom_err_vec(const size_t i,
                            const char* function,
                            const T& y,
                            const char* name,
                            const char* error_msg,
                            const T_msg error_msg2,
                            T_result* result) {
      std::ostringstream msg_o;
      msg_o << name << "[" << stan::error_index::value + i << "] " << error_msg << error_msg2;

      std::string msg;
      msg += (boost::format(function) % typeid(typename T::value_type).name()).str();
      msg += ": ";
      msg += msg_o.str();
      
      throw std::domain_error((boost::format(msg) % stan::get(y,i)).str());
      return false;
    }
    
  }
}
#endif
