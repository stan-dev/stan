#ifndef STAN_LANG_AST_NODE_POSITIVE_ORDERED_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_POSITIVE_ORDERED_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/positive_ordered_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold the declaration of a positive ordered vector. 
     */
    struct positive_ordered_block_var_decl : public var_decl {
      /**
       * Type object specifies size and optional bounds.
       */
      positive_ordered_block_type type_;

      /**
       * Construct a default positive ordered vector declaration.
       */
      positive_ordered_block_var_decl();

      /**
       * Construct a positive ordered vector declaration with the specified
       * name and type.
       *
       * @param name variable name
       * @param type variable type
       */
      positive_ordered_block_var_decl(const std::string& name,
                                      const positive_ordered_block_type& type);

      /**
       * Construct a positive ordered vector declaration with the specified
       * name, type, and definition.
       *
       * @param name variable name
       * @param type variable type
       * @param def defition of variable
       */
      positive_ordered_block_var_decl(const std::string& name,
                                      const positive_ordered_block_type& type,
                                      const expression& def);
    };
  }
}
#endif
