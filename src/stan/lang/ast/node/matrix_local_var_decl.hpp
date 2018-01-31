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
       * name and type.
       *
       * @param name variable name
       * @param type variable type
       */
      matrix_local_var_decl(const std::string& name,
                            const matrix_local_type& type);

      /**
       * Construct a matrix local variable declaration with the specified
       * name, type, and definition.
       *
       * @param name variable name
       * @param type variable type
       * @param def definition
       */
      matrix_local_var_decl(const std::string& name,
                            const matrix_local_type& type,
                            const expression& def);
    };
  }
}
#endif
