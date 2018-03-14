#ifndef STAN_LANG_AST_MATRIX_TYPE_HPP
#define STAN_LANG_AST_MATRIX_TYPE_HPP

#include <string>

namespace stan {
  namespace lang {

    /**
     * Matrix type.
     */
    struct matrix_type {
      /**
       * True if variable type declared with "data" qualifier.
       */
      bool is_data_;

      /**
       * Construct a matrix type with default values.
       */
      matrix_type();

      /**
       * Construct a matrix type with the specified data-only variable flag.
       *
       * @param bool data-only flag
       */
      matrix_type(bool is_data);  // NOLINT(runtime/explicit)

      /**
       * Returns identity string for this type.
       */
      std::string oid() const;
    };

  }
}
#endif
