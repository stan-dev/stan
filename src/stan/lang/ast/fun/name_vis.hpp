#ifndef STAN_LANG_AST_FUN_NAME_VIS_HPP
#define STAN_LANG_AST_FUN_NAME_VIS_HPP

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
     * A visitor for the variant type of variable declarations that
     * returns the name of the variable.
     */
    struct name_vis : public boost::static_visitor<std::string> {
      /**
       * Construct a name visitor.
       */
      name_vis();

      /**
       * Return the empty string.
       *
       * @param x variable declaration
       * @return the empty string
       */
      std::string operator()(const nil& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const int_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const double_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const vector_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const row_vector_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const matrix_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const simplex_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const unit_vector_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const ordered_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const positive_ordered_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const cholesky_factor_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const cholesky_corr_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const cov_matrix_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const corr_matrix_var_decl& x) const;
    };

  }
}
#endif
