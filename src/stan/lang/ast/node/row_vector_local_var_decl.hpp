#ifndef STAN_LANG_AST_NODE_ROW_VECTOR_LOCAL_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_ROW_VECTOR_LOCAL_VAR_DECL_HPP

#include <stan/lang/ast/type/row_vector_local_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a row vector variable declaration.
     */
    struct row_vector_local_var_decl : public var_decl {
      /**
       * Type object specifies size.
       */
      row_vector_local_type type_;

      /**
       * Construct a row vector variable declaration with default
       * values.
       */
      row_vector_local_var_decl();

      /**
       * Construct a row vector with the specified name, size, and definition.
       * Definition is nil if var isn't initialized via declaration.
       *
       * @param name variable name
       * @param N row vector length
       * @param def defition of variable
       */
      row_vector_local_var_decl(const std::string& name,
                                const expression& N,
                                const expression& def);
    };
  }
}
#endif
