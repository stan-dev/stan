#ifndef STAN__MATRIX__CHECK_RANGE_HPP
#define STAN__MATRIX__CHECK_RANGE_HPP

#include <sstream> 
#include <stdexcept>
#include <stan/error_handling/out_of_range.hpp>
#include <stan/meta/traits.hpp>

namespace stan {
  namespace error_handling {

    /**
     * Checks to see if an index into a container is 
     * within range. If the index is out of range, throws
     * an <code>out_of_range</code> exception with the
     * specified message.
     *
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
     * Checks to see if an index into a container is 
     * within range. If the index is out of range, throws
     * an <code>out_of_range</code> exception with the
     * specified message.
     *
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
     * Checks to see if an index into a container is 
     * within range. If the index is out of range, throws
     * an <code>out_of_range</code> exception with the
     * specified message.
     *
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
