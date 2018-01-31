#ifndef STAN_LANG_AST_NODE_ORDERED_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_ORDERED_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/ordered_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold the declaration of an ordered vector. 
     */
    struct ordered_block_var_decl : public var_decl {
      /**
       * Type object specifies size and optional bounds.
       */
      ordered_block_type type_;

      /**
       * Construct a default ordered vector declaration.
       */
      ordered_block_var_decl();

      /**
       * Construct an ordered vector declaration with
       * the specified name and type.
       *
       * @param name variable name
       * @param type variable type
       */
      ordered_block_var_decl(const std::string& name,
                             const ordered_block_type& type);

      /**
       * Construct an ordered vector declaration with
       * the specified name, type, and definition.
       *
       * @param name variable name
       * @param type variable type
       * @param def defition of variable
       */
      ordered_block_var_decl(const std::string& name,
                             const ordered_block_type& type,
                             const expression& def);
    };
  }
}
#endif
