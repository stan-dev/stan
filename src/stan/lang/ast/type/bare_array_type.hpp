#ifndef STAN_LANG_AST_BARE_ARRAY_TYPE_HPP
#define STAN_LANG_AST_BARE_ARRAY_TYPE_HPP

#include <string>

namespace stan {
  namespace lang {

    struct bare_expr_type;
    struct double_type;
    struct int_type;
    struct matrix_type;
    struct row_vector_type;
    struct vector_type;
    
    /** 
     * Bare array type for Stan variables and expressions (recursive).
     */
    struct bare_array_type {
      /**
       * The array element type.
       */
      bare_expr_type element_type_;
      /**
       * Construct an array local var type with default values.
       */
      bare_array_type();

      /**
       * Construct a bare array type with the specified element type.
       *
       * @param el_type element type 
       */
      bare_array_type(const bare_expr_type& el_type);  // NOLINT(runtime/explicit)

      /**
       * Returns type of elements stored in innermost array.
       */
      bare_expr_type contains() const;

      /**
       * Returns number of array dimensions.
       */
      int dims() const;

      /**
       * Returns identity string for this type.
       */
      std::string oid() const;

    };
  }
}
#endif
