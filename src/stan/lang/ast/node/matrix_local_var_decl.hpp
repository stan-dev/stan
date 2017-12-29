#ifndef STAN_LANG_AST_NODE_MATRIX_LOCAL_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_MATRIX_LOCAL_VAR_DECL_HPP

#include <stan/lang/ast/type/matrix_local_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a matrix local variable declaration.
     */
    struct matrix_local_var_decl : public var_decl {
      /**
       * Type object specifies rows, columns.
       */
      matrix_local_type type_;

      /**
       * Construct a matrix local variable declaration with default values.
       */
      matrix_local_var_decl();

      /**
       * Construct a matrix local variable declaration with the specified
       * name, number of rows, number of columns, and definition.
       *
       * @param name variable name
       * @param M number of rows
       * @param N number of columns
       * @param def definition
       */
      matrix_local_var_decl(const std::string& name,
                            const expression& M,
                            const expression& N,
                            const expression& def);
    };
  }
}
#endif
