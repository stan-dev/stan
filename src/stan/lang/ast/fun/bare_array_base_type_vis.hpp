#ifndef STAN_LANG_AST_FUN_BARE_ARRAY_BASE_TYPE_VIS_HPP
#define STAN_LANG_AST_FUN_BARE_ARRAY_BASE_TYPE_VIS_HPP

#include <stan/lang/ast/type/bare_array_type.hpp>
#include <stan/lang/ast/type/double_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_type.hpp>
#include <stan/lang/ast/type/matrix_type.hpp>
#include <stan/lang/ast/type/row_vector_type.hpp>
#include <stan/lang/ast/type/vector_type.hpp>
#include <boost/variant/static_visitor.hpp>

namespace stan {
  namespace lang {

    /**
     * Visitor to get base type from array type.
     */
    struct bare_array_base_type_vis : public boost::static_visitor<bare_expr_type> {
      /**
       * Construct a visitor.
       */
      bare_array_base_type_vis();

      /**
       * Return base type held by array type.
       *
       * @param x type
       * @return base type
       */
      bare_expr_type operator()(const bare_array_type& x) const;

      /**
       * Return base type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      bare_expr_type operator()(const double_type& x) const;

      /**
       * Return base type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      bare_expr_type operator()(const ill_formed_type& x) const;

      /**
       * Return base type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      bare_expr_type operator()(const int_type& x) const;

      /**
       * Return base type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      bare_expr_type operator()(const matrix_type& x) const;

      /**
       * Return base type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      bare_expr_type operator()(const row_vector_type& x) const;

      /**
       * Return base type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      bare_expr_type operator()(const vector_type& x) const;


      /**
       * Return base type held by array type.
       *
       * @param x type
       * @return ill_formed_type
       */
      bare_expr_type operator()(const void_type& x) const;
    };
  }
}
#endif
