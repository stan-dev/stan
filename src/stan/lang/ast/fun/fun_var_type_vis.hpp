#ifndef STAN_LANG_AST_FUN_FUN_VAR_TYPE_VIS_HPP
#define STAN_LANG_AST_FUN_FUN_VAR_TYPE_VIS_HPP

#include <stan/lang/ast/nil.hpp>
#include <stan/lang/ast/type/bare_expr_type.hpp>
#include <stan/lang/ast/node/array_fun_var_decl.hpp>
#include <stan/lang/ast/node/double_fun_var_decl.hpp>
#include <stan/lang/ast/node/int_fun_var_decl.hpp>
#include <stan/lang/ast/node/matrix_fun_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_fun_var_decl.hpp>
#include <stan/lang/ast/node/vector_fun_var_decl.hpp>
#include <boost/variant/static_visitor.hpp>


namespace stan {
  namespace lang {

    /**
     * A visitor which returns the corresponding fun_var_type
     * for this fun_var_decl.
     */
    struct fun_var_type_vis : public boost::static_visitor<fun_var_type> {
      /**
       * Construct a fun_var_type visitor.
       */
      fun_var_type_vis();

      /**
       * Return the corresponding fun_var_type
       *
       * @param x variable declaration
       * @return fun_var_type
       */
      fun_var_type operator()(const nil& x) const;

      /**
       * Return the corresponding fun_var_type_
       *
       * @param x variable declaration
       * @return fun_var_type
       */
      fun_var_type operator()(const array_fun_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return fun_var_type_
       */
      fun_var_type operator()(const int_fun_var_decl& x) const;

      /**
       * Return the corresponding fun_var_type_
       *
       * @param x variable declaration
       * @return fun_var_type_
       */
      fun_var_type operator()(const double_fun_var_decl& x) const;

      /**
       * Return the corresponding fun_var_type_
       *
       * @param x variable declaration
       * @return fun_var_type_
       */
      fun_var_type operator()(const vector_fun_var_decl& x) const;

      /**
       * Return the corresponding fun_var_type_
       *
       * @param x variable declaration
       * @return  the corresponding fun_var_type_
       */
      fun_var_type operator()(const row_vector_fun_var_decl& x) const;

      /**
       * Returnthe corresponding fun_var_type_
       *
       * @param x variable declaration
       * @return fun_var_type_
       */
      fun_var_type operator()(const matrix_fun_var_decl& x) const;
    };

  }
}
#endif
