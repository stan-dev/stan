#ifndef STAN_LANG_AST_NODE_FUN_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_FUN_VAR_DECL_HPP

#include <stan/lang/ast/node/var_decl.hpp>
#include <stan/lang/ast/type/bare_expr_type.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Structure to hold a function argument declaration.
     */
    struct fun_var_decl : public var_decl {

      /**
       * The line in the source code where the declaration begins.
       */
      std::size_t begin_line_;

      /**
       * The line in the source code where the declaration ends.
       */
      std::size_t end_line_;

      /**
       * Specific type of this variable.
       */
      bare_expr_type type_;

      /**
       * Construct a default variable declaration.
       */
      fun_var_decl();

      /**
       * Construct a fun variable declaration with the specified
       * name and type.
       *
       * @param name variable name
       * @param type variable type
       */
      fun_var_decl(const std::string& name,
                     const bare_expr_type& type);

      /**
       * Construct a fun variable declaration with the specified
       * name, type, and definition.
       *
       * @param name variable name
       * @param tyep variable type
       * @param def definition
       */
      fun_var_decl(const std::string& name,
                     const bare_expr_type& type,
                     const expression& def);


      /**
       * Return the variable declaration's bare expr type.
       *
       * @return the bare expr type
       */
      bare_expr_type bare_type() const;

      /**
       * Return the variable declaration's definition.
       *
       * @return expression definition for this variable
       */
      expression def() const;

      /**
       * Return true if variable declaration contains a definition.
       *
       * @return bool indicating has or doesn't have definition
       */
      bool has_def() const;

      /**
       * Return the variable declaration's name.
       *
       * @return name of variable
       */
      std::string name() const;

      /**
       * Return the variable declaration's bare_expr_type
       * which contains size specifications.
       *
       * @return bare_expr_type
       */
      bare_expr_type type() const;
    };

  }
}
#endif
