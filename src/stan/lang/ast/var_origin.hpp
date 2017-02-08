#ifndef STAN_LANG_AST_VAR_ORIGIN_HPP
#define STAN_LANG_AST_VAR_ORIGIN_HPP


#include <stan/lang/ast/origin_block.hpp>
#include <cstddef>

namespace stan {
  namespace lang {

    /**
     * Structure which tracks enclosing program block(s) encountered by parser.
     * Var_map records program block where variable declared.
     * Grammar rules check allowed constructs in (enclosing) block.
     */
    struct var_origin {
      /**
       * Outermost enclosing program block.
       */
      origin_block program_block_;

      /**
       * True if in a nested (local) program block.
       */
      bool is_local_;


      /**
       * No arg constructor, defaults:
       * program_block_ : model_name_origin
       * is_local : false
       */
      var_origin();

      /**
       * Construct an origin for variable in a specified block
       * is_local : false
       *
       * @param program_block enclosing program block
       */
      var_origin(const
                 origin_block& program_block);   // NOLINT(runtime/explicit)

      /**
       * Construct an origin for a variable in specified outer program block,
       * specify whether or not variable is in local program block,
       * all other bool flags false
       *
       * @param program_block enclosing program block
       * @param is_local flags whether or not in a local block
       */
      var_origin(const origin_block& program_block,
                 const bool& is_local);

      /**
       * Return true when declared in void_function_argument_origin block.
       *
       * @return true for void function origin block types
       */
      bool is_void_function_origin() const;

      /**
       * Return true when enclosing block is void function type
       *
       * @return true for void function origin block types
       */
      bool is_non_void_function_origin() const;

      /**
       * Return true when program block allows access to LP
       * i.e., model block or lp function
       *
       * @return true when program block allows access to LP
       */
      bool allows_lp() const;

      /**
       * Return true when program block allows access to RNG
       * i.e., transformed data block or rng function
       *
       * @return true when program block allows access to RNG
       */
      bool allows_rng() const;
    };

  }
}
#endif
