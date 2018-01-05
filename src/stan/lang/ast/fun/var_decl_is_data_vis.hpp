#ifndef STAN_LANG_AST_FUN_VAR_DECL_IS_DATA_VIS_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_IS_DATA_VIS_HPP

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
     * A visitor for the variant type of fun_var_decl the
     * returns the value of member is_data_
     */
    struct var_decl_is_data_vis : public boost::static_visitor<bool> {
      /**
       * Construct a var_decl_type visitor.
       */
      var_decl_is_data_vis();

      /**
       * Return false.
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const nil& x) const;

      /**
       * Return is_data_
       *
       * @param x variable declaration
       * @return is_data_
       */
      bool operator()(const array_fun_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return is_data_
       */
      bool operator()(const int_fun_var_decl& x) const;

      /**
       * Return is_data_
       *
       * @param x variable declaration
       * @return is_data_
       */
      bool operator()(const double_fun_var_decl& x) const;

      /**
       * Return is_data_
       *
       * @param x variable declaration
       * @return is_data_
       */
      bool operator()(const vector_fun_var_decl& x) const;

      /**
       * Return is_data_
       *
       * @param x variable declaration
       * @return is_data_
       */
      bool operator()(const row_vector_fun_var_decl& x) const;

      /**
       * Return is_data_
       *
       * @param x variable declaration
       * @return is_data_
       */
      bool operator()(const matrix_fun_var_decl& x) const;
    };

  }
}
#endif
