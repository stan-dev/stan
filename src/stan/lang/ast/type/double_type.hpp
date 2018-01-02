#ifndef STAN_LANG_AST_DOUBLE_TYPE_HPP
#define STAN_LANG_AST_DOUBLE_TYPE_HPP

#include <string>

namespace stan {
  namespace lang {

    /**
     * Double type.
     */
    struct double_type {
      /**
       * Returns identity string for this type.
       */
      std::string oid() const;
    };

  }
}
#endif
