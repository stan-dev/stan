#ifndef STAN__ERROR_HANDLING__MATRIX__CHECK_SIZE_MATCH_HPP
#define STAN__ERROR_HANDLING__MATRIX__CHECK_SIZE_MATCH_HPP

#include <sstream>
#include <boost/type_traits/common_type.hpp>
#include <stan/error_handling/scalar/dom_err.hpp>

namespace stan {
  namespace error_handling {

    // FIXME: update warnings
    template <typename T_size1, typename T_size2>
    inline bool check_size_match(const std::string& function,
                                 const std::string& name_i,
                                 T_size1 i,
                                 const std::string& name_j, 
                                 T_size2 j) {
      typedef typename boost::common_type<T_size1,T_size2>::type common_type;
      if (static_cast<common_type>(i) == static_cast<common_type>(j))
        return true;

      std::ostringstream msg;
      msg << ") and " 
          << name_j << " (" << j << ") must match in size";
      dom_err(function, name_i, i,
              "(", msg.str());
      return false;
    }

  }
}
#endif
