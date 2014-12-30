#ifndef STAN__ERROR_HANDLING__OUT_OF_RANGE_HPP
#define STAN__ERROR_HANDLING__OUT_OF_RANGE_HPP

#include <typeinfo>
#include <string>
#include <sstream>
#include <stdexcept>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Throw an out_of_range exception with a consistently formatted message.
     * 
     * This is an abstraction for all Stan functions to use when throwing
     * out of range. This will allow us to change the behavior for all
     * functions at once.
     *
     * The message is:
     * "<function>: index <index> out of range; expecting index to be between "
     * "1 and <max> for <name>.<msg1><msg2>"
     *
     * @param function Name of the function
     * @param name Name of the variable
     * @param max Max
     * @param index Index
     * @param msg1 Message to print
     * @param msg2 Message to print
     */
    inline void out_of_range(const std::string& function,
                             const std::string& name,
                             const int max,
                             const int index,
                             const std::string& msg1,
                             const std::string& msg2) {
      std::ostringstream message;
      
      message << function << ": accessing element out of range. "
              << "index " << index << " out of range; "
              << "expecting index to be between "
              << stan::error_index::value << " and " 
              << stan::error_index::value - 1 + max
              << " for " << name << "."
              << msg1
              << msg2;

      throw std::out_of_range(message.str());
    }

    /**
     * Throw an out_of_range exception with a consistently formatted message.
     * 
     * This is an abstraction for all Stan functions to use when throwing
     * out of range. This will allow us to change the behavior for all
     * functions at once. (We've already changed behavior mulitple times up
     * to Stan v2.5.0.)
     *
     * The message is:
     * "<function>: index <index> out of range; expecting index to be between "
     * "1 and <max> for <name>.<msg1>"
     *
     * @param function Name of the function
     * @param name Name of the variable
     * @param max Max
     * @param index Index
     * @param msg1 Message to print
     */
    inline void out_of_range(const std::string& function,
                             const std::string& name,
                             const int max,
                             const int index,
                             const std::string& msg1) {
      out_of_range(function, name, max, index, msg1, "");
    }


    /**
     * Throw an out_of_range exception with a consistently formatted message.
     * 
     * This is an abstraction for all Stan functions to use when throwing
     * out of range. This will allow us to change the behavior for all
     * functions at once. (We've already changed behavior mulitple times up
     * to Stan v2.5.0.)
     *
     * The message is:
     * "<function>: index <index> out of range; expecting index to be between "
     * "1 and <max> for <name>."
     *
     * @param function Name of the function
     * @param name Name of the variable
     * @param index Index
     * @param max Max
     */
    inline void out_of_range(const std::string& function,
                             const std::string& name,
                             const int max,
                             const int index) {
      out_of_range(function, name, max, index, "", "");
    }

  }
}
#endif
