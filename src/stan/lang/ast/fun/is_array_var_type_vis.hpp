#ifndef STAN_LANG_AST_FUN_IS_ARRAY_VAR_TYPE_VIS_HPP
#define STAN_LANG_AST_FUN_IS_ARRAY_VAR_TYPE_VIS_HPP

#include <stan/lang/ast/type/array_bare_type.hpp>
#include <stan/lang/ast/type/array_block_type.hpp>
#include <stan/lang/ast/type/array_local_type.hpp>
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
     * Visitor to check if type is an array type
     */
    struct is_array_var_type_vis : public boost::static_visitor<bool> {
      /**
       * Construct the array type check visitor.
       */
      is_array_var_type_vis();

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return true
       */
      bool operator()(const array_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return true
       */
      bool operator()(const array_local_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return true
       */
      bool operator()(const array_bare_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const cholesky_corr_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const cholesky_factor_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const corr_matrix_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const cov_matrix_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const double_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const double_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const ill_formed_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const int_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const int_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const matrix_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const matrix_local_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const matrix_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const ordered_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const positive_ordered_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const row_vector_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const row_vector_local_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const row_vector_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const simplex_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const unit_vector_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const vector_block_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const vector_local_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const vector_type& x) const;

      /**
       * Return true if the specified type is an array type.
       *
       * @param x type
       * @return false
       */
      bool operator()(const void_type& x) const;
    };
  }
}
#endif
