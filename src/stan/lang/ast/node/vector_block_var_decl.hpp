#ifndef STAN_LANG_AST_NODE_VECTOR_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_VECTOR_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/vector_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a column vector variable declaration.
     */
    struct vector_block_var_decl : public var_decl {
      /**
       * Type object specifies size.
       */
      vector_block_type type_;

      /**
       * Construct a column vector variable declaration with default
       * values.
       */
      vector_block_var_decl();

      /**
       * Construct a column vector declaration with the specified
       * name, bounds, number of rows, and definition.
       * Definition is nil if var isn't initialized via declaration.
       *
       * @param name variable name
       * @param bounds variable upper and/or lower bounds
       * @param N number of rows
       * @param def defition of variable
       */
      vector_block_var_decl(const std::string& name,
                            const range& bounds,
                            const expression& N,
                            const expression& def);
    };
  }
}
#endif
