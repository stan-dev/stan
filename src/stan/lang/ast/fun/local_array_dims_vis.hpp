#ifndef STAN_LANG_AST_FUN_LOCAL_ARRAY_DIMS_VIS_HPP
#define STAN_LANG_AST_FUN_LOCAL_ARRAY_DIMS_VIS_HPP

#include <stan/lang/ast/type/local_array_type.hpp>
#include <stan/lang/ast/type/double_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_type.hpp>
#include <stan/lang/ast/type/matrix_local_type.hpp>
#include <stan/lang/ast/type/row_vector_local_type.hpp>
#include <stan/lang/ast/type/vector_local_type.hpp>
#include <boost/variant/static_visitor.hpp>

namespace stan {
  namespace lang {

    /**
     * Visitor to get array dimensions from array type.
     */
    struct local_array_dims_vis : public boost::static_visitor<int> {
      /**
       * Construct a visitor.
       */
      local_array_dims_vis();

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return number of array dimensions
       */
      int operator()(const local_array_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const double_type& x) const;

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
      int operator()(const int_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const matrix_local_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const row_vector_local_type& x) const;

      /**
       * Return number of array dimensions for this type.
       *
       * @param x type
       * @return 0
       */
      int operator()(const vector_local_type& x) const;
    };
  }
}
#endif
