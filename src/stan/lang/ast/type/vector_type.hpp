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
       * Returns identity string for this type.
       */
      std::string oid() const;
    };

  }
}
#endif
