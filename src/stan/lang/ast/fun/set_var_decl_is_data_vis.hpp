#ifndef STAN_LANG_AST_FUN_SET_VAR_DECL_IS_DATA_VIS_HPP
#define STAN_LANG_AST_FUN_SET_VAR_DECL_IS_DATA_VIS_HPP

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

namespace stan {
  namespace lang {

    /**
     * A visitor for the variant type of variable declarations that
     * sets `is_data_` to `true` for the corresponding var_decl object.
     */
    struct set_var_decl_is_data_vis : public boost::static_visitor<bool> {
      /**
       * Construct a set_var_decl_is_data visitor.
       */
      set_var_decl_is_data_vis();

      /**
       * Do nothing.
       */
      bool operator()(const nil& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(array_block_var_decl& x) const;
      bool operator()(array_local_var_decl& x) const;
      bool operator()(array_fun_var_decl& x) const;

      bool operator()(int_block_var_decl& x) const;
      bool operator()(int_local_var_decl& x) const;
      bool operator()(int_fun_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(double_block_var_decl& x) const;
      bool operator()(double_local_var_decl& x) const;
      bool operator()(double_fun_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(vector_block_var_decl& x) const;
      bool operator()(vector_local_var_decl& x) const;
      bool operator()(vector_fun_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(row_vector_block_var_decl& x) const;
      bool operator()(row_vector_local_var_decl& x) const;
      bool operator()(row_vector_fun_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(matrix_block_var_decl& x) const;
      bool operator()(matrix_local_var_decl& x) const;
      bool operator()(matrix_fun_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(cholesky_factor_block_var_decl& x) const;

      /**
       * Return the bool for this variable.
       */
      bool operator()(cholesky_corr_block_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(cov_matrix_block_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(corr_matrix_block_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(ordered_block_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(positive_ordered_block_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(simplex_block_var_decl& x) const;

      /**
       * Set `is_data_` to `true` for variable's `var_decl` object.
       */
      bool operator()(unit_vector_block_var_decl& x) const;
    };

  }
}
#endif
