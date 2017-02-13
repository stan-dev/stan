#ifndef STAN_LANG_AST_SCOPE_HPP
#define STAN_LANG_AST_SCOPE_HPP


#include <stan/lang/ast/origin_block.hpp>
#include <cstddef>

namespace stan {
  namespace lang {

    /**
     * Structure which tracks enclosing program block(s) encountered by parser.
     * Var_map records program block where variable declared.
     * Grammar rules check allowed constructs in (enclosing) block.
     */
    struct scope {
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
      scope();

      /**
       * Construct an origin for variable in a specified block
       * is_local : false
       *
       * @param program_block enclosing program block
       */
      scope(const
                 origin_block& program_block);   // NOLINT(runtime/explicit)

      /**
       * Construct an origin for a variable in specified outer program block,
       * specify whether or not variable is in local program block,
       * all other bool flags false
       *
       * @param program_block enclosing program block
       * @param is_local flags whether or not in a local block
       */
      scope(const origin_block& program_block,
                 const bool& is_local);

      /**
       * Return true when declared in data block.
       *
       * @return true for data origin block types
       */
      bool is_data_origin() const;

      /**
       * Return true when declared in top-level of parameter or
       * transformed parameter block.
       *
       * @return true for parameter origin block types
       */
      bool is_parameter_origin() const;

      /**
       * Return true when declared in top-level parameter block
       *
       * @return true for top-level parameter block
       */
      bool is_non_local_parameter_origin() const;

      /**
       * Return true when declared in top-level transformed parameter block
       *
       * @return true for top-level transformed parameter block
       */
      bool is_non_local_transformed_parameter_origin() const;
      /**
       * Return false when declared in transformed parameter block
       * or local block.
       *
       * @return true for non-parameter block types
       */
      bool is_non_parameter_origin() const;

      /**
       * Return true when declared in any function argument block.
       *
       * @return true for function origin block types
       */
      bool is_fun_origin() const;

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
       * Return true when program block allows assignment to variables
       * i.e., not data or parameter block
       *
       * @return true when program block allows access to LP
       */
      bool allows_assignment() const;

      /**
       * Return true when program block allows access to LP function
       *
       * @return true when program block allows access to LP function
       */
      bool allows_lp_fun() const;

      /**
       * Return true when program block allows access to LP statement
       *
       * @return true when program block allows access to LP statement
       */
      bool allows_lp_stmt() const;

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
