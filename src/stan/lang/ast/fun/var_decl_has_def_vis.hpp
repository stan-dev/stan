#ifndef STAN_LANG_AST_FUN_VAR_DECL_HAS_DEF_VIS_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_HAS_DEF_VIS_HPP


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
#include <stan/lang/ast/node/int_block_var_decl.hpp>
#include <stan/lang/ast/node/int_local_var_decl.hpp>
#include <stan/lang/ast/node/matrix_block_var_decl.hpp>
#include <stan/lang/ast/node/matrix_local_var_decl.hpp>
#include <stan/lang/ast/node/ordered_block_var_decl.hpp>
#include <stan/lang/ast/node/positive_ordered_block_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_block_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_local_var_decl.hpp>
#include <stan/lang/ast/node/simplex_block_var_decl.hpp>
#include <stan/lang/ast/node/unit_vector_block_var_decl.hpp>
#include <stan/lang/ast/node/vector_block_var_decl.hpp>
#include <stan/lang/ast/node/vector_local_var_decl.hpp>
#include <boost/variant/static_visitor.hpp>

namespace stan {
  namespace lang {
    
    /**
     * Visitor to check whether are variable declaration contains a definition.
     */
    struct var_decl_has_def_vis : public boost::static_visitor<bool> {
      /**
       * Construct a var_decl_has_def visitor.
       */
      var_decl_has_def_vis();

      /**
       * Returns true if variable declaration has a definition.
       *
       * @param x variable declaration
       */
      bool operator()(const nil& x) const;

      /**
       * Returns true if variable declaration has a definition.
       *
       * @param x variable declaration
       */
      bool operator()(const array_block_var_decl& x) const;
      bool operator()(const array_local_var_decl& x) const;

      bool operator()(const cholesky_factor_block_var_decl& x) const;
      bool operator()(const cholesky_corr_block_var_decl& x) const;
      bool operator()(const cov_matrix_block_var_decl& x) const;
      bool operator()(const corr_matrix_block_var_decl& x) const;

      /**
       * Returns true if variable declaration has a definition.
       *
       * @param x variable declaration
       */
      bool operator()(const double_block_var_decl& x) const;
      bool operator()(const double_local_var_decl& x) const;

      /**
       * Return the definition for a variable declaration.
       *
       * @param x variable declaration
       */
      bool operator()(const int_block_var_decl& x) const;
      bool operator()(const int_local_var_decl& x) const;

      /**
       * Returns true if variable declaration has a definition.
       *
       * @param x variable declaration
       */
      bool operator()(const matrix_block_var_decl& x) const;
      bool operator()(const matrix_local_var_decl& x) const;

      bool operator()(const ordered_block_var_decl& x) const;
      bool operator()(const positive_ordered_block_var_decl& x) const;

      /**
       * Returns true if variable declaration has a definition.
       *
       * @param x variable declaration
       */
      bool operator()(const row_vector_block_var_decl& x) const;
      bool operator()(const row_vector_local_var_decl& x) const;

      bool operator()(const simplex_block_var_decl& x) const;
      bool operator()(const unit_vector_block_var_decl& x) const;

      /**
       * Returns true if variable declaration has a definition.
       *
       * @param x variable declaration
       */
      bool operator()(const vector_block_var_decl& x) const;
      bool operator()(const vector_local_var_decl& x) const;
    };
  }
}
#endif
