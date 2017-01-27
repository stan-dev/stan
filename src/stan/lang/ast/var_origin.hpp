#ifndef STAN_LANG_AST_VAR_ORIGIN_HPP
#define STAN_LANG_AST_VAR_ORIGIN_HPP


#include <stan/lang/ast/origin_block.hpp>
#include <cstddef>

namespace stan {
  namespace lang {

    /**
     * Structure describes enclosing program block(s)
     * in which variable is declared.
     */
    struct var_origin {
      /**
       * Outermost enclosing program block.
       */
      origin_block program_block_;

      /**
       * Flags whether in a nested (local) program block.
       */
      bool is_local_;

      /**
       * Construct an empty var_origin
       */
      var_origin();

      /**
       * Construct an origin for variable in a specified block
       *
       * @param program_block - enclosing program block
       */
      var_origin(const origin_block program_block);  // NOLINT(runtime/explicit)

      /**
       * Construct an origin for a variable in specified outer program block,
       * specify whether or not variable is in local program block 
       *
       * @param program_block - enclosing program block
       * @param is_local - flags whether or not in a local block
       */
      var_origin(const origin_block program_block,
                 const bool is_local);  // NOLINT(runtime/explicit)

      /**
       * Return true if declared in a local block.
       *
       * @return bool if declared in a local block.
       */
      bool is_local() const;
    };

  }
}
#endif
