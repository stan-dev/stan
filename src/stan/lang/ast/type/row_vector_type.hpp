#ifndef STAN_LANG_AST_ROW_VECTOR_TYPE_HPP
#define STAN_LANG_AST_ROW_VECTOR_TYPE_HPP

#include <string>

namespace stan {
  namespace lang {

    /**
     * Row vector type.
     */
    struct row_vector_type {
      /**
       * Returns identity string for this type.
       */
      std::string oid() const;
    };

  }
}
#endif
