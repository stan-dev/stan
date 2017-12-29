#ifndef STAN_LANG_AST_NODE_ROW_VECTOR_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_ROW_VECTOR_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/row_vector_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a row vector variable declaration.
     */
    struct row_vector_block_var_decl : public var_decl {
      /**
       * Type object specifies size and optional bounds.
       */
      row_vector_block_type type_;

      /**
       * Construct a row vector variable declaration with default
       * values.
       */
      row_vector_block_var_decl();

      /**
       * Construct a row vector declaration with the specified
       * name, bounds, length, and definition.
       * Definition is nil if var isn't initialized via declaration.
       *
       * @param name variable name
       * @param bounds variable upper and/or lower bounds
       * @param N row vector length
       * @param def defition of variable
       */
      row_vector_block_var_decl(const std::string& name,
                                const range& bounds,
                                const expression& N,
                                const expression& def);
    };
  }
}
#endif
