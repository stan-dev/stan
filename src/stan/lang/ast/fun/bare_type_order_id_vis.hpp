#ifndef STAN_LANG_AST_FUN_BARE_TYPE_ORDER_ID_VIS_HPP
#define STAN_LANG_AST_FUN_BARE_TYPE_ORDER_ID_VIS_HPP

#include <stan/lang/ast/type/bare_array_type.hpp>
#include <stan/lang/ast/type/double_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_type.hpp>
#include <stan/lang/ast/type/matrix_type.hpp>
#include <stan/lang/ast/type/row_vector_type.hpp>
#include <stan/lang/ast/type/vector_type.hpp>
#include <boost/variant/static_visitor.hpp>
#include <string>
namespace stan {
  namespace lang {

    /**
     * Visitor to get order id string for variant bare_expr_type.
     */
    struct bare_type_order_id_vis : public boost::static_visitor<std::string> {
      /**
       * Construct a visitor.
       */
      bare_type_order_id_vis();

      /**
       * Return identity string for this type.
       *
       * @param x type
       * @return identity string
       */
      std::string operator()(const bare_array_type& x) const;

      /**
       * Return identity string for this type.
       *
       * @param x type
       * @return identity string
       */
      std::string operator()(const double_type& x) const;

      /**
       * Return identity string for this type.
       *
       * @param x type
       * @return identity string
       */
      std::string operator()(const ill_formed_type& x) const;

      /**
       * Return identity string for this type.
       *
       * @param x type
       * @return identity string
       */
      std::string operator()(const int_type& x) const;

      /**
       * Return identity string for this type.
       *
       * @param x type
       * @return identity string
       */
      std::string operator()(const matrix_type& x) const;

      /**
       * Return identity string for this type.
       *
       * @param x type
       * @return identity string
       */
      std::string operator()(const row_vector_type& x) const;

      /**
       * Return identity string for this type.
       *
       * @param x type
       * @return identity string
       */
      std::string operator()(const vector_type& x) const;

      /**
       * Return identity string for this type.
       *
       * @param x type
       * @return identity string
       */
      std::string operator()(const void_type& x) const;
    };
  }
}
#endif
