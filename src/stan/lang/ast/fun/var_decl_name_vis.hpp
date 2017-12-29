#ifndef STAN_LANG_AST_FUN_VAR_DECL_NAME_VIS_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_NAME_VIS_HPP

#include <stan/lang/ast/nil.hpp>
#include <stan/lang/ast/node/array_block_var_decl.hpp>
#include <stan/lang/ast/node/array_fun_var_decl.hpp>
#include <stan/lang/ast/node/array_local_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_corr_block_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_factor_block_var_decl.hpp>
#include <stan/lang/ast/node/corr_matrix_block_var_decl.hpp>
#include <stan/lang/ast/node/cov_matrix_block_var_decl.hpp>
#include <stan/lang/ast/node/double_block_var_decl.hpp>
#include <stan/lang/ast/node/double_local_var_decl.hpp>
#include <stan/lang/ast/node/double_fun_var_decl.hpp>
#include <stan/lang/ast/node/int_block_var_decl.hpp>
#include <stan/lang/ast/node/int_local_var_decl.hpp>
#include <stan/lang/ast/node/int_fun_var_decl.hpp>
#include <stan/lang/ast/node/matrix_block_var_decl.hpp>
#include <stan/lang/ast/node/matrix_local_var_decl.hpp>
#include <stan/lang/ast/node/matrix_fun_var_decl.hpp>
#include <stan/lang/ast/node/ordered_block_var_decl.hpp>
#include <stan/lang/ast/node/positive_ordered_block_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_block_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_local_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_fun_var_decl.hpp>
#include <stan/lang/ast/node/simplex_block_var_decl.hpp>
#include <stan/lang/ast/node/unit_vector_block_var_decl.hpp>
#include <stan/lang/ast/node/vector_block_var_decl.hpp>
#include <stan/lang/ast/node/vector_local_var_decl.hpp>
#include <stan/lang/ast/node/vector_fun_var_decl.hpp>
#include <boost/variant/static_visitor.hpp>
#include <string>

namespace stan {
  namespace lang {

    /**
     * A visitor for the variant type of variable declarations that
     * returns the name of the variable.
     */
    struct var_decl_name_vis : public boost::static_visitor<std::string> {
      /**
       * Construct a var_decl_name visitor.
       */
      var_decl_name_vis();

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
      std::string operator()(const array_block_var_decl& x) const;
      std::string operator()(const array_local_var_decl& x) const;
      std::string operator()(const array_fun_var_decl& x) const;

      std::string operator()(const int_block_var_decl& x) const;
      std::string operator()(const int_local_var_decl& x) const;
      std::string operator()(const int_fun_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const double_block_var_decl& x) const;
      std::string operator()(const double_local_var_decl& x) const;
      std::string operator()(const double_fun_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const vector_block_var_decl& x) const;
      std::string operator()(const vector_local_var_decl& x) const;
      std::string operator()(const vector_fun_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const row_vector_block_var_decl& x) const;
      std::string operator()(const row_vector_local_var_decl& x) const;
      std::string operator()(const row_vector_fun_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const matrix_block_var_decl& x) const;
      std::string operator()(const matrix_local_var_decl& x) const;
      std::string operator()(const matrix_fun_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const cholesky_factor_block_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const cholesky_corr_block_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const cov_matrix_block_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const corr_matrix_block_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const ordered_block_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const positive_ordered_block_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const simplex_block_var_decl& x) const;

      /**
       * Return the name of the variable.
       *
       * @param x variable declaration
       * @return the name of the variable being declared
       */
      std::string operator()(const unit_vector_block_var_decl& x) const;
    };

  }
}
#endif
