#ifndef STAN_LANG_AST_NODE_UNIT_VECTOR_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_UNIT_VECTOR_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/unit_vector_block_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold the declaration of a unit vector. 
     */
    struct unit_vector_block_var_decl : public var_decl {
      /**
       * Type object specifies size.
       */
      unit_vector_block_type type_;

      /**
       * Construct a default unit vector declaration.
       */
      unit_vector_block_var_decl();

      /**
       * Construct a unit vector declaration with
       * the specified name and size.
       *
       * @param name variable name
       * @param K vector size
       */
      unit_vector_block_var_decl(const std::string& name,
                                 const expression& K);

      /**
       * Construct a unit vector declaration with
       * the specified name, size, and definition.
       * Definition is nil if var isn't initialized via declaration.
       *
       * @param name variable name
       * @param K vector size
       * @param def defition of variable
       */
      unit_vector_block_var_decl(const std::string& name,
                                 const expression& K,
                                 const expression& def);
    };
  }
}
#endif
