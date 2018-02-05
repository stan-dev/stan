#ifndef STAN_LANG_AST_FUN_BLOCK_ARRAY_DIMS_VIS_HPP
#define STAN_LANG_AST_FUN_BLOCK_ARRAY_DIMS_VIS_HPP

#include <stan/lang/ast/type/block_array_type.hpp>
#include <stan/lang/ast/type/cholesky_factor_corr_block_type.hpp>
#include <stan/lang/ast/type/cholesky_factor_cov_block_type.hpp>
#include <stan/lang/ast/type/corr_matrix_block_type.hpp>
#include <stan/lang/ast/type/cov_matrix_block_type.hpp>
#include <stan/lang/ast/type/double_block_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_block_type.hpp>
#include <stan/lang/ast/type/matrix_block_type.hpp>
#include <stan/lang/ast/type/ordered_block_type.hpp>
#include <stan/lang/ast/type/positive_ordered_block_type.hpp>
#include <stan/lang/ast/type/row_vector_block_type.hpp>
#include <stan/lang/ast/type/simplex_block_type.hpp>
#include <stan/lang/ast/type/unit_vector_block_type.hpp>
#include <stan/lang/ast/type/vector_block_type.hpp>
#include <boost/variant/static_visitor.hpp>

namespace stan {
  namespace lang {

    /**
     * Visitor to get array dimensions from array type.
     */
    struct block_array_dims_vis : public boost::static_visitor<int> {
      /**
       * Construct a visitor.
       */
      block_array_dims_vis();

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return number of array dimensions
       */
      int operator()(const block_array_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const cholesky_factor_corr_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const cholesky_factor_cov_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const corr_matrix_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const cov_matrix_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const double_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const ill_formed_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const int_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const matrix_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const ordered_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const positive_ordered_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const row_vector_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const simplex_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const unit_vector_block_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const vector_block_type& x) const;
    };
  }
}
#endif
