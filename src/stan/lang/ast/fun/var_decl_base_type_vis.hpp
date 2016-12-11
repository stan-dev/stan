#ifndef STAN_LANG_AST_FUN_VAR_DECL_BASE_TYPE_VIS_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_BASE_TYPE_VIS_HPP

#include <stan/lang/ast/node/cholesky_corr_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_factor_var_decl.hpp>
#include <stan/lang/ast/node/corr_matrix_var_decl.hpp>
#include <stan/lang/ast/node/cov_matrix_var_decl.hpp>
#include <stan/lang/ast/node/double_var_decl.hpp>
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

namespace stan {
  namespace lang {

    /**
     * Visitor to extract base variable declaration from the variant
     * types of a variable declaration.
     */
    struct var_decl_base_type_vis
      : public boost::static_visitor<base_var_decl> {
      /**
       * Construct a variable declaration visitor.
       */
      var_decl_base_type_vis();

      /**
       * Return the nil variable declaration for the specified nil
       * variable declaration.
       *
       * @param x variable declaration
       * @return nil variable declaration
       */
      base_var_decl operator()(const nil& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const int_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const double_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const vector_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const row_vector_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const matrix_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const simplex_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const unit_vector_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const ordered_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const positive_ordered_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const cholesky_factor_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const cholesky_corr_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const cov_matrix_var_decl& x) const;

      /**
       * Return the base variable declaration for the specified
       * variable declaration.
       *
       * @param x variable declaration
       * @return base declaration
       */
      base_var_decl operator()(const corr_matrix_var_decl& x) const;
    };

  }
}
#endif
