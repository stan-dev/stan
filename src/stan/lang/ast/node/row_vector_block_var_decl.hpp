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
       * name and type.
       *
       * @param name variable name
       * @param type variable type
       */
      row_vector_block_var_decl(const std::string& name,
                                const row_vector_block_type& type);

      /**
       * Construct a row vector declaration with the specified
       * name, type, and definition.
       *
       * @param name variable name
       * @param type variable type
       * @param def defition of variable
       */
      row_vector_block_var_decl(const std::string& name,
                                const row_vector_block_type& type,
                                const expression& def);
    };
  }
}
#endif
