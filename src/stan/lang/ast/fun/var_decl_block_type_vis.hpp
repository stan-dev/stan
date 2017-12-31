#ifndef STAN_LANG_AST_FUN_VAR_DECL_BLOCK_TYPE_VIS_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_BLOCK_TYPE_VIS_HPP

#include <stan/lang/ast/nil.hpp>
#include <stan/lang/ast/type/block_var_type.hpp>
#include <stan/lang/ast/node/array_block_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_corr_block_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_factor_block_var_decl.hpp>
#include <stan/lang/ast/node/corr_matrix_block_var_decl.hpp>
#include <stan/lang/ast/node/cov_matrix_block_var_decl.hpp>
#include <stan/lang/ast/node/double_block_var_decl.hpp>
#include <stan/lang/ast/node/int_block_var_decl.hpp>
#include <stan/lang/ast/node/matrix_block_var_decl.hpp>
#include <stan/lang/ast/node/ordered_block_var_decl.hpp>
#include <stan/lang/ast/node/positive_ordered_block_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_block_var_decl.hpp>
#include <stan/lang/ast/node/simplex_block_var_decl.hpp>
#include <stan/lang/ast/node/unit_vector_block_var_decl.hpp>
#include <stan/lang/ast/node/vector_block_var_decl.hpp>
#include <boost/variant/static_visitor.hpp>


namespace stan {
  namespace lang {

    /**
     * A visitor to get block_var_type from block_var_decls.
     */
    struct var_decl_block_type_vis
      : public boost::static_visitor<block_var_type> {
      /**
       * Construct a var_decl_type visitor.
       */
      var_decl_block_type_vis();

      /**
       * Return the ill-formed type
       *
       * @param x variable declaration
       * @return ill_formed_type
       */
      block_var_type operator()(const nil& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const array_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const int_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const double_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const vector_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const row_vector_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const matrix_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const cholesky_factor_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const cholesky_corr_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const cov_matrix_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const corr_matrix_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const ordered_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const positive_ordered_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const simplex_block_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      block_var_type operator()(const unit_vector_block_var_decl& x) const;
    };

  }
}
#endif
