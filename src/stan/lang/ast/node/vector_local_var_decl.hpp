#ifndef STAN_LANG_AST_NODE_VECTOR_LOCAL_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_VECTOR_LOCAL_VAR_DECL_HPP

#include <stan/lang/ast/type/vector_local_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a column vector local variable declaration.
     */
    struct vector_local_var_decl : public var_decl {
      /**
       * Type object specifies size.
       */
      vector_local_type type_;

      /**
       * Construct a column vector variable local variable declaration
       * with default values.
       */
      vector_local_var_decl();

      /**
       * Construct a column vector declaration with
       * the specified name and number of rows.
       *
       * @param name variable name
       * @param N number of rows
       */
      vector_local_var_decl(const std::string& name,
                            const expression& N);

      /**
       * Construct a column vector declaration with
       * the specified name, number of rows, and definition.
       * Definition is nil if var isn't initialized via declaration.
       *
       * @param name variable name
       * @param N number of rows
       * @param def defition of variable
       */
      vector_local_var_decl(const std::string& name,
                            const expression& N,
                            const expression& def);
    };
  }
}
#endif
