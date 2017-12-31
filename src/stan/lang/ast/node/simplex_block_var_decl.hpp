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
       * the specified name and size.
       *
       * @param name variable name
       * @param K simplex size
       */
      simplex_block_var_decl(const std::string& name,
                             const expression& K);

      /**
       * Construct a simplex declaration with
       * the specified name, size, and definition.
       * Definition is nil if var isn't initialized via declaration.
       *
       * @param name variable name
       * @param K simplex size
       * @param def defition of variable
       */
      simplex_block_var_decl(const std::string& name,
                             const expression& K,
                             const expression& def);
    };
  }
}
#endif
