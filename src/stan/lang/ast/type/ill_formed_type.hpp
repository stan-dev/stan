#ifndef STAN_LANG_AST_ILL_FORMED_TYPE_HPP
#define STAN_LANG_AST_ILL_FORMED_TYPE_HPP

#include <string>

namespace stan {
  namespace lang {

    /**
     * Ill_Formed type.
     */
    struct ill_formed_type {
      /**
       * Returns identity string for this type.
       */
      std::string oid() const;
    };

  }
}
#endif

