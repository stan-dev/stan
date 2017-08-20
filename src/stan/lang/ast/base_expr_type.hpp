#ifndef STAN_LANG_AST_BASE_EXPR_TYPE_HPP
#define STAN_LANG_AST_BASE_EXPR_TYPE_HPP

#include <boost/variant/recursive_variant.hpp>

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
      base_expr_type_t;

      base_expr_type();
      base_expr_type(const base_expr_type_t&
                     base_type);  // NOLINT(runtime/explicit)

      base_expr_type(const ill_formed_type&
                     base_type);  // NOLINT(runtime/explicit)
      base_expr_type(const void_type&
                     base_type);  // NOLINT(runtime/explicit)
      base_expr_type(const int_type&
                     base_type);  // NOLINT(runtime/explicit)
      base_expr_type(const double_type&
                     base_type);  // NOLINT(runtime/explicit)
      base_expr_type(const vector_type&
                     base_type);  // NOLINT(runtime/explicit)
      base_expr_type(const row_vector_type&
                     base_type);  // NOLINT(runtime/explicit)
      base_expr_type(const matrix_type&
                     base_type);  // NOLINT(runtime/explicit)


      bool operator==(const base_expr_type& base_type) const;

      bool is_ill_formed_type() const;
      bool is_void_type() const;
      bool is_int_type() const;
      bool is_double_type() const;
      bool is_vector_type() const;
      bool is_row_vector_type() const;
      bool is_matrix_type() const;

      /**
       * The base expr type held by this wrapper.
       */
      base_expr_type_t base_type_;
    };
  }
}
#endif
