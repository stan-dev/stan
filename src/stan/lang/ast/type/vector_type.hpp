#ifndef STAN_LANG_AST_VECTOR_TYPE_HPP
#define STAN_LANG_AST_VECTOR_TYPE_HPP

#include <string>

namespace stan {
  namespace lang {

    /**
     * Vector type.
     */
    struct vector_type {
      /**
       * True if variable type declared with "data" qualifier.
       */
      bool is_data_;

      /**
       * Construct a vector type with default values.
       */
      vector_type();

      /**
       * Construct a vector type with the specified data-only variable flag.
       *
       * @param bool data-only flag
       */
      vector_type(bool is_data);  // NOLINT(runtime/explicit)

      /**
       * Returns identity string for this type.
       */
      std::string oid() const;
    };

  }
}
#endif
