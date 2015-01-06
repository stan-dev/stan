#ifndef STAN__MATRIX__CHECK_RANGE_HPP
#define STAN__MATRIX__CHECK_RANGE_HPP

#include <sstream> 
#include <stdexcept>
#include <stan/error_handling/out_of_range.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Return <code>true</code> if specified index is within range.
     *
     * This check is 1-indexed by default. This behavior can be
     * changed by setting <code>stan::error_index::value</code>.
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param max Maximum size of the variable
     * @param index Index to check
     * @param nested_level Nested level (for error messages)
     * @param error_msg Additional error message (for error messages)
     * 
     * @return <code>true</code> if the index is within range
     * @throw <code>std::out_of_range</code> if the index is not in range
     */
    inline bool check_range(const std::string& function,
                            const std::string& name,  
                            const size_t max,
                            const size_t index,
                            const size_t nested_level,
                            const std::string& error_msg) {
      if ((index >= stan::error_index::value) 
          && (index < max + stan::error_index::value))
        return true;
      
      std::stringstream msg;
      msg << "; index position = " << nested_level;
      
      out_of_range(function, max, index, msg.str(), error_msg);
      return false;
    }

    /**
     * Return <code>true</code> if specified index is within range.
     *
     * This check is 1-indexed by default. This behavior can be
     * changed by setting <code>stan::error_index::value</code>.
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param max Maximum size of the variable
     * @param index Index to check
     * @param error_msg Additional error message (for error messages)
     * 
     * @return <code>true</code> if the index is within range
     * @throw <code>std::out_of_range</code> if the index is not in range
     */
    inline bool check_range(const std::string& function,
                            const std::string& name,  
                            const size_t max,
                            const size_t index,
                            const std::string& error_msg) {
      if ((index >= stan::error_index::value) 
          && (index < max + stan::error_index::value))
        return true;
      
      out_of_range(function, max, index, error_msg);
      return false;
    }

    /**
     * Return <code>true</code> if specified index is within range.
     *
     * This check is 1-indexed by default. This behavior can be
     * changed by setting <code>stan::error_index::value</code>.
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param max Maximum size of the variable
     * @param index Index to check
     * 
     * @return <code>true</code> if the index is within range
     * @throw <code>std::out_of_range</code> if the index is not in range
     */
    inline bool check_range(const std::string& function,
                            const std::string& name,  
                            const size_t max,
                            const size_t index) {
      if ((index >= stan::error_index::value) 
          && (index < max + stan::error_index::value))
        return true;
      
      out_of_range(function, max, index);
      return false;
    }


  }
}
#endif
