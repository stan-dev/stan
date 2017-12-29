#ifndef STAN_LANG_AST_NODE_CHOLESKY_FACTOR_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_CHOLESKY_FACTOR_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/cholesky_factor_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a Cholesky factor variable declaration.
     */
    struct cholesky_factor_block_var_decl : public var_decl {
      /**
       * Type object specifies number of rows, columns.
       */
      cholesky_factor_block_type type_;

      /**
       * Construct a variable declaration for a Cholesky factor 
       * with default values.
       */
      cholesky_factor_block_var_decl();

      /**
       * Construct a Cholesky factor variable declaration with the
       * specified name, number of rows, number of columns, and definition.
       * Definition is nil if var isn't initialized via declaration.
       *
       * @param name variable name
       * @param M number of rows
       * @param N number of columns
       * @param def defition of variable
       */
      cholesky_factor_block_var_decl(const std::string& name,
                                     const expression& M,
                                     const expression& N,
                                     const expression& def);
    };
  }
}
#endif
