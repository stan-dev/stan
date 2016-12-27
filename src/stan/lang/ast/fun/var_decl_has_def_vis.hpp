#ifndef STAN_LANG_AST_FUN_VAR_DECL_HAS_DEF_VIS_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_HAS_DEF_VIS_HPP

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
     * Variable declaration visitor functor for determining if a
     * variable declaration includes a definition.
     */
    struct var_decl_has_def_vis : public boost::static_visitor<bool> {
      /**
       * Construct the visitor.
       */
      var_decl_has_def_vis();

      /**
       * Return true if the specified variable declaration includes a
       * variable definition (always false for the nil declaration). 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const nil& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const int_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const double_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const vector_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const row_vector_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const matrix_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const simplex_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const unit_vector_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const ordered_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const positive_ordered_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const cholesky_factor_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const cholesky_corr_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const cov_matrix_var_decl& x) const;

      /**
       * Return true if the specified variable declaration includes a
       * variable definition. 
       *
       * @param x variable declaration
       * @return false
       */
      bool operator()(const corr_matrix_var_decl& x) const;
    };

  }
}
#endif
