#ifndef __STAN__MATH__ERROR_HANDLING__DOM_ERR_HPP__
#define __STAN__MATH__ERROR_HANDLING__DOM_ERR_HPP__

#include <sstream>
#include <boost/math/policies/policy.hpp>
#include <stan/math/error_handling/raise_domain_error.hpp>

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


    template <typename T_y, 
              typename T_result,
              typename T_msg2>
    inline bool dom_err(const char* function,
                        const T_y& y,
                        const char* name,
                        const char* error_msg,
                        T_msg2 error_msg2,
                        T_result* result) {
      using stan::math::policies::raise_domain_error;
      std::ostringstream msg_o;
      msg_o << name << error_msg << error_msg2;
      std::string msg(msg_o.str());
      T_result tmp = raise_domain_error<T_result,T_y>(function,
                                                      msg.c_str(),
                                                      y,
                                                      boost::math::policies::policy<>());
      if (result != 0)
        *result = tmp;
      return false;
    }
    
  }
}
#endif
