#ifndef STAN_LANG_AST_NODE_MATRIX_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_MATRIX_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/matrix_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a matrix block variable declaration.
     */
    struct matrix_block_var_decl : public var_decl {
      /**
       * Type object specifies rows, columns, and bounds (optional). 
       */
      matrix_block_type type_;

      /**
       * Construct a matrix block variable declaration with default values.
       */
      matrix_block_var_decl();

      /**
       * Construct a matrix block variable declaration with the specified
       * name, bounds, number of rows, number of columns, and definition.
       *
       * @param name variable name
       * @param bounds variable upper and/or lower bounds
       * @param M number of rows
       * @param N number of columns
       * @param def definition
       */
      matrix_block_var_decl(const std::string& name,
                            const range& bounds,
                            const expression& M,
                            const expression& N,
                            const expression& def);
    };
  }
}
#endif
