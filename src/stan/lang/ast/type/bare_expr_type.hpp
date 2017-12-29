#ifndef STAN_LANG_AST_BARE_EXPR_TYPE_HPP
#define STAN_LANG_AST_BARE_EXPR_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <string>

namespace stan {
  namespace lang {

    /** 
     * Bare type for Stan variables and expressions.
     */
    struct array_bare_type;
    struct double_type;
    struct ill_formed_type;
    struct int_type;
    struct matrix_type;
    struct row_vector_type;
    struct vector_type;
    struct void_type;

    struct bare_expr_type {
      static const int ORDER_ID = 0;

      /**
       * Recursive wrapper for bare types.
       */
      typedef boost::variant<
        boost::recursive_wrapper<ill_formed_type>,
        boost::recursive_wrapper<double_type>,
        boost::recursive_wrapper<int_type>,
        boost::recursive_wrapper<matrix_type>,
        boost::recursive_wrapper<row_vector_type>,
        boost::recursive_wrapper<vector_type>,
        boost::recursive_wrapper<void_type>,
        boost::recursive_wrapper<array_bare_type> >
      bare_t;

      /**
       * The bare type held by this wrapper.
       */
      bare_t bare_type_;

      /**
       * Fixed numerical ID used for sorting.
       */
      int order_id_;

      /**
       * Construct a bare var type with default values.
       */
      bare_expr_type();

      /**
       * Construct a bare var type with the specified variant type.
       *
       * @param type bare type raw variant type.
       */
      bare_expr_type(const bare_expr_type& type);  // NOLINT(runtime/explicit)

      /**
       * Construct a bare type with the specified type.
       *
       * @param type bare type
       */      
      bare_expr_type(const ill_formed_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a bare type with the specified type.
       *
       * @param type bare type
       */      
      bare_expr_type(const double_type& type); // NOLINT(runtime/explicit)
      /**
       * Construct a bare type with the specified type.
       *
       * @param type bare type
       */      

      bare_expr_type(const int_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a bare type with the specified type.
       *
       * @param type bare type
       */      
      bare_expr_type(const matrix_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a bare type with the specified type.
       *
       * @param type bare type
       */      
      bare_expr_type(const row_vector_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a bare type with the specified type.
       *
       * @param type bare type
       */      
      bare_expr_type(const vector_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a bare type with the specified type.
       *
       * @param type bare type
       */      
      bare_expr_type(const void_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a bare type with the specified type.
       *
       * @param type bare type
       */      
      bare_expr_type(const array_bare_type& type); // NOLINT(runtime/explicit)

      /**
       * Return true if the specified bare type is the same as
       * this bare type.
       *
       * @param bare_type Other bare type.
       * @return result of equality test.
       */
      bool operator==(const bare_expr_type& bare_type) const;

      /**
       * Return true if the specified bare type is not the same as
       * this bare type.
       *
       * @param bare_type Other bare type.
       * @return result of inequality test.
       */
      bool operator!=(const bare_expr_type& bare_type) const;

      /**
       * Return true if this bare type `order_id_` 
       * is less than that of the specified bare type.
       *
       * @param bare_type Other bare type.
       * @return result of comparison.
       */
      bool operator<(const bare_expr_type& bare_type) const;

      /**
       * Return true if this bare type `order_id_` 
       * is less than or equal to that of the specified bare type.
       *
       * @param bare_type Other bare type.
       * @return result of comparison.
       */
      bool operator<=(const bare_expr_type& bare_type) const;

      /**
       * Return true if this bare type `order_id_` 
       * is greater than that of the specified bare type.
       *
       * @param bare_type Other bare type.
       * @return result of comparison.
       */
      bool operator>(const bare_expr_type& bare_type) const;

      /**
       * Return true if this bare type `order_id_` 
       * is greater than or equal to that of the specified bare type.
       *
       * @param bare_type Other bare type.
       * @return result of comparison.
       */
      bool operator>=(const bare_expr_type& bare_type) const;

      /**
       * Returns true if `bare_type_` is `ill_formed_type`, false otherwise.
       */
      bool is_ill_formed_type() const;

      /**
       * Returns true if `bare_type_` is `void_type`, false otherwise.
       */
      bool is_void_type() const;

      /**
       * Returns true if `bare_type_` is `int_type`, false otherwise.
       */
      bool is_int_type() const;

      /**
       * Returns true if `bare_type_` is `double_type`, false otherwise.
       */
      bool is_double_type() const;

      /**
       * Returns true if `bare_type_` is `vector_type`, false otherwise.
       */
      bool is_vector_type() const;

      /**
       * Returns true if `bare_type_` is `row_vector_type`, false otherwise.
       */
      bool is_row_vector_type() const;

      /**
       * Returns true if `bare_type_` is `matrix_type`, false otherwise.
       */
      bool is_matrix_type() const;

      /**
       * Returns true if `bare_type_` is `array_bare_type`, false otherwise.
       */
      bool is_array_var_type() const;

      /**
       * Returns array element type if `var_type_` is `array_bare_type`,
       * ill_formed_type otherwise.  (Call `is_array_var_type()` first.)
       */
      bare_expr_type get_array_el_type() const;

      /**
       * Returns total number of dimensions for container type.
       * Returns 0 for scalar types.
       */
      int num_dims() const;
    };
  }
}
#endif
