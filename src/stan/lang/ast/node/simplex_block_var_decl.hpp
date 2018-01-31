#ifndef STAN_LANG_AST_NODE_SIMPLEX_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_SIMPLEX_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/simplex_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold the declaration of a simplex. 
     */
    struct simplex_block_var_decl : public var_decl {
      /**
       * Type object specifies size.
       */
      simplex_block_type type_;

      /**
       * Construct a default simplex declaration.
       */
      simplex_block_var_decl();

      /**
       * Construct a simplex declaration with
       * the specified name and type.
       *
       * @param name variable name
       * @param type variable type
       */
      simplex_block_var_decl(const std::string& name,
                             const simplex_block_type& type);

      /**
       * Construct a simplex declaration with
       * the specified name, type, and definition.
       *
       * @param name variable name
       * @param type variable type
       * @param def defition of variable
       */
      simplex_block_var_decl(const std::string& name,
                             const simplex_block_type& type,
                             const expression& def);
    };
  }
}
#endif
