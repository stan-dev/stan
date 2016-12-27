#ifndef STAN_LANG_AST_FUN_VAR_DECL_DIMS_VIS_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_DIMS_VIS_HPP

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
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Structure for visitor to extract the dimension sizes in a
     * variable declaration variant type.
     */
    struct var_decl_dims_vis
      : public boost::static_visitor<std::vector<expression> > {
      /**
       * Construt a dimensions visitor for variable declarations.
       */
      var_decl_dims_vis();

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration (here, the empty vector).
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const nil& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const int_var_decl& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const double_var_decl& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const vector_var_decl& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const row_vector_var_decl& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const matrix_var_decl& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const simplex_var_decl& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const unit_vector_var_decl& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const ordered_var_decl& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(
                                  const positive_ordered_var_decl& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(
                                  const cholesky_factor_var_decl& x) const;
      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */

      std::vector<expression> operator()(const cholesky_corr_var_decl& x) const;
      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const cov_matrix_var_decl& x) const;

      /**
       * Return the sequence of dimension size expressions for the
       * specified variable declaration.
       *
       * @param x variable declaration
       * @return sequence of dimension sizes
       */
      std::vector<expression> operator()(const corr_matrix_var_decl& x) const;
    };

  }
}
#endif
