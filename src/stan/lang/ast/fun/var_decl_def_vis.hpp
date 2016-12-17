#ifndef STAN_LANG_AST_FUN_VAR_DECL_DEF_VIS_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_DEF_VIS_HPP

#include <stan/lang/ast/node/cholesky_corr_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_factor_var_decl.hpp>
#include <stan/lang/ast/node/corr_matrix_var_decl.hpp>
#include <stan/lang/ast/node/cov_matrix_var_decl.hpp>
#include <stan/lang/ast/node/double_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/int_var_decl.hpp>
#include <stan/lang/ast/node/matrix_var_decl.hpp>
#include <stan/lang/ast/node/nil.hpp>
#include <stan/lang/ast/node/ordered_var_decl.hpp>
#include <stan/lang/ast/node/positive_ordered_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_var_decl.hpp>
#include <stan/lang/ast/node/simplex_var_decl.hpp>
#include <stan/lang/ast/node/unit_vector_var_decl.hpp>
#include <stan/lang/ast/node/vector_var_decl.hpp>
#include <boost/variant/static_visitor.hpp>

namespace stan {
  namespace lang {

    /**
     * Visitor to return the definition in a variable declaration.  
     */
    struct var_decl_def_vis
      : public boost::static_visitor<expression> {
      /**
       * Construct a variable declaration definition visitor.
       */
      var_decl_def_vis();

      /**
       * Return the definition for the specified variable
       * declaration (in this case nil for nil input).
       *
       * @param x variable declaration
       */
      expression operator()(const nil& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const int_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const double_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const vector_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const row_vector_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const matrix_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const simplex_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const unit_vector_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const ordered_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const positive_ordered_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const cholesky_factor_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const cholesky_corr_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const cov_matrix_var_decl& x) const;

      /**
       * Return the definition for the specified variable
       * declaration.
       *
       * @param x variable declaration
       */
      expression operator()(const corr_matrix_var_decl& x) const;
    };

  }
}
#endif
