#ifndef STAN__IO__JSON__JSON_ERROR_HPP
#define STAN__IO__JSON__JSON_ERROR_HPP

#include <stdexcept>


namespace stan {

  namespace json {

    /**
     * Exception type for JSON errors.
     */
    struct json_error : public std::logic_error {
      /**
       * Construct a JSON error with the specified message
       * @param what Message to attach to error
       */
      json_error(const std::string& what)
        : logic_error(what) {
      }

    };


  }
}
#endif
