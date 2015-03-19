#ifndef STAN__MATH__PRIM__SCAL__ERR__DOMAIN_ERROR_VEC_HPP
#define STAN__MATH__PRIM__SCAL__ERR__DOMAIN_ERROR_VEC_HPP

#include <sstream>
#include <string>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/scal/meta/value_type.hpp>
#include <stan/math/prim/scal/meta/error_index.hpp>
#include <stan/math/prim/scal/meta/get.hpp>

namespace stan {

  namespace math {

    /**
     * Throw a domain error with a consistently formatted message.
     *
     * This is an abstraction for all Stan functions to use when throwing
     * domain errors. This will allow us to change the behavior for all
     * functions at once. (We've already changed behavior mulitple times up
     * to Stan v2.5.0.)
     *
     * The message is:
     * "<function>: <name>[<i+error_index>] <msg1><y>"
     *    where error_index is the value of stan::error_index::value
     * which indicates whether the message should be 0 or 1 indexed.
     *
     * @tparam T Type of variable
     * @param function Name of the function
     * @param name Name of the variable
     * @param y Variable
     * @param i Index
     * @param msg1 Message to print before the variable
     * @param msg2 Message to print after the variable
     * @throw std::domain_error
     */
    template <typename T>
    inline void domain_error_vec(const char* function,
                                 const char* name,
                                 const T& y,
                                 const size_t i,
                                 const char* msg1,
                                 const char* msg2) {
      std::ostringstream vec_name_stream;
      vec_name_stream << name
                      << "[" << stan::error_index::value + i << "]";
      std::string vec_name(vec_name_stream.str());
      domain_error(function, vec_name.c_str(), stan::get(y, i) , msg1, msg2);
    }

    /**
     * Throw a domain error with a consistently formatted message.
     *
     * This is an abstraction for all Stan functions to use when throwing
     * domain errors. This will allow us to change the behavior for all
     * functions at once. (We've already changed behavior mulitple times up
     * to Stan v2.5.0.)
     *
     * The message is:
     * "<function>: <name>[<i+error_index>] <msg1><y>"
     *   where error_index is the value of stan::error_index::value
     * which indicates whether the message should be 0 or 1 indexed.
     *
     * @tparam T Type of variable
     * @param function Name of the function
     * @param name Name of the variable
     * @param y Variable
     * @param i Index
     * @param msg Message to print before the variable
     * @throw std::domain_error
     */
    template <typename T>
    inline void domain_error_vec(const char* function,
                                 const char* name,
                                 const T& y,
                                 const size_t i,
                                 const char* msg) {
      domain_error_vec(function, name, y, i, msg, "");
    }

  }
}
#endif
