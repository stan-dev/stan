#ifndef STAN_LANG_AST_NODE_LOCAL_ARRAY_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_LOCAL_ARRAY_VAR_DECL_HPP

#include <stan/lang/ast/type/local_var_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * An array local variable declaration and optional definition.
     */
    struct array_local_var_decl : public var_decl {
      /**
       * Array type defines element type, array size.
       */
      local_array_type type_;

      /**
       * Construct an array local variable declaration with default
       * values. 
       */
      array_local_var_decl();

      /**
       * Construct an array variable declaration with the specified
       * name, element type, and length.
       *
       * @param name variable name
       * @param el_type element type
       * @param len array length
       */
      array_local_var_decl(const std::string& name,
                           const local_var_type& el_type,
                           const expression& len);

      /**
       * Construct an array variable declaration with the specified
       * name, element type, length, and definition.
       *
       * @param name variable name
       * @param el_type element type
       * @param len array length
       * @param def definition
       */
      array_local_var_decl(const std::string& name,
                           const local_var_type& el_type,
                           const expression& len,
                           const expression& def);
    };
  }
}
#endif
