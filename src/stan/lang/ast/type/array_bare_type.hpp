#ifndef STAN_LANG_AST_ARRAY_BARE_TYPE_HPP
#define STAN_LANG_AST_ARRAY_BARE_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <string>

namespace stan {
  namespace lang {

    /** 
     * Bare array type for Stan variables and expressions (recursive).
     */
    struct array_bare_type {
      /**
       * The array element type.
       */
      bare_expr_type element_type_;
      /**
       * Construct an array local var type with default values.
       */
      array_bare_type();

      /**
       * Construct a bare array type with the specified element type.
       *
       * @param el_type element type 
       */
      array_bare_type(const bare_expr_type& el_type);  // NOLINT(runtime/explicit)

      /**
       * Returns type of elements stored in innermost array.
       */
      bare_expr_type contains() const;

      /**
       * Returns number of array dimensions.
       */
      int array_dims() const;
    };
  }
}
#endif
