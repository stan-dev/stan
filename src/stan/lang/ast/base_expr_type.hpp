#ifndef STAN_LANG_AST_BASE_EXPR_TYPE_HPP
#define STAN_LANG_AST_BASE_EXPR_TYPE_HPP

#include <boost/variant/recursive_variant.hpp>
#include <cstddef>

namespace stan {
  namespace lang {

    struct ill_formed_type;
    struct void_type;
    struct int_type;
    struct double_type;
    struct vector_type;
    struct row_vector_type;
    struct matrix_type;

    /**
     * Struct to wrap the variant base type of expressions.
     */
    struct base_expr_type {
      static const int ORDER_ID = 0;

      /**
       * The variant base type of expressions.
       */
      typedef boost::variant<
        boost::recursive_wrapper<ill_formed_type>,
        boost::recursive_wrapper<void_type>,
        boost::recursive_wrapper<int_type>,
        boost::recursive_wrapper<double_type>,
        boost::recursive_wrapper<vector_type>,
        boost::recursive_wrapper<row_vector_type>,
        boost::recursive_wrapper<matrix_type> >
      base_t;

      /**
       * The base expr type held by this wrapper.
       */
      base_t base_type_;

      /**
       * Fixed numerical ID used for sorting.
       */
      int order_id_;

      /**
       * Construct an empty base type.
       */
      base_expr_type();

      /**
       * Construct a base expression type with the specified base type.
       *
       * @param base_type base type
       */
      base_expr_type(const base_expr_type& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a base expression type from from a variant base type.
       *
       * @param base_type base type
       */
      template <typename T>
      base_expr_type(const T& base_type);  // NOLINT(runtime/explicit)

      /**
       * Return true if the specified base expression type is the same as
       * this base expression type.
       *
       * @param base_type Other base type.
       * @return result of equality test.
       */
      bool operator==(const base_expr_type& base_type) const;

      /**
       * Return true if the specified base expression type is not the same as
       * this base expression type.
       *
       * @param base_type Other base type.
       * @return result of inequality test.
       */
      bool operator!=(const base_expr_type& base_type) const;

      /**
       * Return true if this base expression type `order_id_` 
       * is less than that of the specified base expression type.
       *
       * @param base_type Other base type.
       * @return result of comparison.
       */
      bool operator<(const base_expr_type& base_type) const;

      /**
       * Return true if this base expression type `order_id_` 
       * is less than or equal to that of the specified base expression type.
       *
       * @param base_type Other base type.
       * @return result of comparison.
       */
      bool operator<=(const base_expr_type& base_type) const;

      /**
       * Return true if this base expression type `order_id_` 
       * is greater than that of the specified base expression type.
       *
       * @param base_type Other base type.
       * @return result of comparison.
       */
      bool operator>(const base_expr_type& base_type) const;

      /**
       * Return true if this base expression type `order_id_` 
       * is greater than or equal to that of the specified base expression type.
       *
       * @param base_type Other base type.
       * @return result of comparison.
       */
      bool operator>=(const base_expr_type& base_type) const;

      /**
       * Returns true if `base_type_` is `ill_formed_type`, false otherwise.
       */
      bool is_ill_formed_type() const;

      /**
       * Returns true if `base_type_` is `void_type`, false otherwise.
       */
      bool is_void_type() const;

      /**
       * Returns true if `base_type_` is `int_type`, false otherwise.
       */
      bool is_int_type() const;

      /**
       * Returns true if `base_type_` is `double_type`, false otherwise.
       */
      bool is_double_type() const;

      /**
       * Returns true if `base_type_` is `vector_type`, false otherwise.
       */
      bool is_vector_type() const;

      /**
       * Returns true if `base_type_` is `row_vector_type`, false otherwise.
       */
      bool is_row_vector_type() const;

      /**
       * Returns true if `base_type_` is `matrix_type`, false otherwise.
       */
      bool is_matrix_type() const;
    };
  }
}
#endif
