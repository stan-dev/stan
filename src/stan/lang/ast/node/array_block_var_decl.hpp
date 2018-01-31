#ifndef STAN_LANG_AST_NODE_ARRAY_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_ARRAY_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/block_array_type.hpp>
#include <stan/lang/ast/type/block_var_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * An array block variable declaration and optional definition.
     */
    struct array_block_var_decl : public var_decl {
      /**
       * Array type defines element type, array size.
       */
      block_array_type type_;

      /**
       * Construct an array block variable declaration with default
       * values. 
       */
      array_block_var_decl();

      /**
       * Construct an array variable declaration with the specified
       * name and type.
       *
       * @param name variable name
       * @param type variable type
       */
      array_block_var_decl(const std::string& name,
                           const block_array_type& type);

      /**
       * Construct an array variable declaration with the specified
       * name, element type, length, and definition.
       *
       * @param name variable name
       * @param type variable type
       * @param def definition
       */
      array_block_var_decl(const std::string& name,
                           const block_array_type& type,
                           const expression& def);
    };
  }
}
#endif
