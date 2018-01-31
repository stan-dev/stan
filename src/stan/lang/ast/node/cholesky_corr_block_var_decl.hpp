#ifndef STAN_LANG_AST_NODE_CHOLESKY_CORR_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_CHOLESKY_CORR_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/cholesky_corr_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a Cholesky factor for a correlation matrix
     * variable declaration.
     */
    struct cholesky_corr_block_var_decl : public var_decl {
      /**
       * Type object specifies number of rows and columns.
       */
      cholesky_corr_block_type type_;

      /**
       * Construct a variable declaration for a Cholesky factor for a
       * correlation matrix.
       */
      cholesky_corr_block_var_decl();

      /**
       * Construct a Cholesky factor variable declaration for a
       * correlation matrix with the specified name and type.
       *
       * @param name variable name
       * @param type variable type
       */
      cholesky_corr_block_var_decl(const std::string& name,
                                   const cholesky_corr_block_type& type);


      /**
       * Construct a Cholesky factor variable declaration for a
       * correlation matrix with the specified name, type, and definition.
       *
       * @param name variable name
       * @param type variable type
       * @param def defition of variable
       */
      cholesky_corr_block_var_decl(const std::string& name,
                                   const cholesky_corr_block_type& type,
                                   const expression& def);
    };
  }
}
#endif
