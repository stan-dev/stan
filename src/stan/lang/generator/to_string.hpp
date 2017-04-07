#ifndef STAN_LANG_GENERATOR_TO_STRING_HPP
#define STAN_LANG_GENERATOR_TO_STRING_HPP

#include <string>
#include <sstream>

namespace stan {
  namespace lang {

    /**
     * Return the string resulting from streaming the specified
     * argument to a default <code>std::stringstream</code>.
     *
     * @tparam T type of primitive or class to convert to string
     * @param[in] x value to convert to string
     * @return string resulting from streaming the value
     */
    template <typename T>
    std::string to_string(const T& x) {
      std::stringstream ss;
      ss << x;
      return ss.str();
    }

  }
}
#endif
