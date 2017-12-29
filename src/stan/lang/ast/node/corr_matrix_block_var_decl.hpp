#ifndef STAN_LANG_AST_NODE_CORR_MATRIX_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_CORR_MATRIX_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/corr_matrix_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a correlation matrix variable declaration.
     */
    struct corr_matrix_block_var_decl : public var_decl {
      /**
       * Type object specifies number of rows and columns.
       */
      corr_matrix_block_type type_;

      /**
       * Construct a variable declaration for a correlation matrix.
       */
      corr_matrix_block_var_decl();

      /**
       * Construct a variable declaration for a correlation matrix
       * with the specified name, size, and definition.
       * Definition is nil if var isn't initialized via declaration.
       *
       * @param name variable name
       * @param K corr matrix size
       * @param def defition of variable
       */
      corr_matrix_block_var_decl(const std::string& name,
                                 const expression& K,
                                 const expression& def);
    };
  }
}
#endif
