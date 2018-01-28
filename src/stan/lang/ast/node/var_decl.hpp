#ifndef STAN_LANG_AST_NODE_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_VAR_DECL_HPP

#include <stan/lang/ast/type/bare_expr_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * AST base class for shared structure of variable declarations.
     */
    struct var_decl {
      /**
       * Variable name.
       */
      std::string name_;

      /**
       * Variable bare type.
       */
      bare_expr_type bare_type_;

      /**
       * True if variable declaration has "data" qualifier.
       */
      bool is_data_;

      /**
       * Definition for variable (nil if undefined).
       */
      expression def_;

     /**
       * Construct a default variable declaration.
       */
      var_decl();

      /**
       * Construct a variable declaration of the specified name.
       *
   
       */
      var_decl(const std::string& name); // NOLINT

      /**
       * Construct a variable declaration with the specified
       * name and type.
       *
       * @param name name of variable
       * @param type bare type of variable
       */
      var_decl(const std::string& name,
               const bare_expr_type& type);

      /**
       * Construct a variable declaration with the specified
       * name, type, and definition.
       *
       * @param name name of variable
       * @param type bare type of variable
       * @param def definition of expression
       */
      var_decl(const std::string& name,
               const bare_expr_type& type,
               const expression& def);

      /**
       * Return var_decl type.
       *
       * @return var_type_
       */
      bare_expr_type bare_type() const;

      /**
       * Return var_decl definition.
       *
       * @return def_
       */
      expression def() const;

      /**
       * Return var_decl is_data_ flag.
       *
       * @return is_data_
       */
      bool is_data() const;

      /**
       * Return var_decl name.
       *
       * @return name_
       */
      std::string name() const;

      /**
       * Set is_data_ flag
       *
       * @param bool true when var must be data_only
       */
      void set_is_data(bool flag);

    };
  }
}
#endif
