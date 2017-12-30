#ifndef STAN_LANG_AST_FUN_GET_TOTAL_DIMS_VIS_HPP
#define STAN_LANG_AST_FUN_GET_TOTAL_DIMS_VIS_HPP

#include <stan/lang/ast/type/bare_array_type.hpp>
#include <stan/lang/ast/type/block_array_type.hpp>
#include <stan/lang/ast/type/local_array_type.hpp>
#include <stan/lang/ast/type/cholesky_corr_block_type.hpp>
#include <stan/lang/ast/type/cholesky_factor_block_type.hpp>
#include <stan/lang/ast/type/corr_matrix_block_type.hpp>
#include <stan/lang/ast/type/cov_matrix_block_type.hpp>
#include <stan/lang/ast/type/double_block_type.hpp>
#include <stan/lang/ast/type/double_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_block_type.hpp>
#include <stan/lang/ast/type/int_type.hpp>
#include <stan/lang/ast/type/matrix_block_type.hpp>
#include <stan/lang/ast/type/matrix_local_type.hpp>
#include <stan/lang/ast/type/matrix_type.hpp>
#include <stan/lang/ast/type/ordered_block_type.hpp>
#include <stan/lang/ast/type/positive_ordered_block_type.hpp>
#include <stan/lang/ast/type/row_vector_block_type.hpp>
#include <stan/lang/ast/type/row_vector_local_type.hpp>
#include <stan/lang/ast/type/row_vector_type.hpp>
#include <stan/lang/ast/type/simplex_block_type.hpp>
#include <stan/lang/ast/type/unit_vector_block_type.hpp>
#include <stan/lang/ast/type/vector_block_type.hpp>
#include <stan/lang/ast/type/vector_local_type.hpp>
#include <stan/lang/ast/type/vector_type.hpp>
#include <stan/lang/ast/type/void_type.hpp>
#include <boost/variant/static_visitor.hpp>

namespace stan {
  namespace lang {
    /**
     * Visitor to count total number of dimensions for a var type.
     * Total is array dimensions and +1 for vectors or +2 for matrices.
     */
    struct get_total_dims_vis : public boost::static_visitor<int> {
      /**
       * Construct a get_total_dims visitor.
       */
      get_total_dims_vis();

      /**
       * Return the number of dimensions for this type.
       *
       * @param x type
       */
      int operator()(const block_array_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const local_array_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const bare_array_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const cholesky_corr_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const cholesky_factor_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const corr_matrix_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const cov_matrix_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const double_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const double_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const ill_formed_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const int_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const int_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const matrix_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const matrix_local_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const matrix_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const ordered_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const positive_ordered_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const row_vector_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const row_vector_local_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const row_vector_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const simplex_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const unit_vector_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const vector_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const vector_local_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const vector_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       */
      int operator()(const void_type& x) const;
    };
  }
}
#endif
