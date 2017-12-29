#ifndef STAN_LANG_AST_FUN_GET_ARRAY_LOCAL_EL_TYPE_VIS_HPP
#define STAN_LANG_AST_FUN_GET_ARRAY_LOCAL_EL_TYPE_VIS_HPP

#include <stan/lang/ast/type/array_local_type.hpp>
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
     * Visitor to get array element type.
     */
    struct get_array_local_el_type_vis : public boost::static_visitor<local_var_type> {
      /**
       * Return element type held by array type.
       *
       * @param x type
       * @return element type
       */
      local_var_type operator()(const array_local_type& x) const;

      /**
       * Return element type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      local_var_type operator()(const double_type& x) const;

      /**
       * Return element type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      local_var_type operator()(const ill_formed_type& x) const;

      /**
       * Return element type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      local_var_type operator()(const int_type& x) const;

      /**
       * Return element type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      local_var_type operator()(const matrix_local_type& x) const;

      /**
       * Return element type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      local_var_type operator()(const row_vector_local_type& x) const;

      /**
       * Return element type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      local_var_type operator()(const vector_local_type& x) const;
    };
  }
}
#endif
