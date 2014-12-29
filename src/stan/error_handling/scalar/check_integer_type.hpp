#ifndef STAN__ERROR_HANDLING__SCALAR__CHECK_INTEGER_TYPE_HPP
#define STAN__ERROR_HANDLING__SCALAR__CHECK_INTEGER_TYPE_HPP

#include <stan/error_handling/scalar/dom_err_vec.hpp>
#include <stan/error_handling/scalar/dom_err.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    // throws if any element in y or low is nan
    template <typename T_n>
    inline bool check_integer_type(const std::string& function,
                                   const std::string& name,
                                   const T_n& n) {
      using boost::is_same;
      using stan::scalar_type;

      if (is_same<scalar_type<T_n>::type,int>)
        return true;

      std::stringstream msg;
      msg << ", but must be of scalar type integer";
      dom_err(function, name, typename scalar_type<T_n>::type,
              "is ", msg.str());

      return false;
    }
  }
}
#endif
