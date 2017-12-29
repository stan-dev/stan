#ifndef STAN_LANG_AST_ARRAY_LOCAL_TYPE_HPP
#define STAN_LANG_AST_ARRAY_LOCAL_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>

namespace stan {
  namespace lang {

    struct local_var_type;

    /** 
     * Local array type for Stan variables and expressions (recursive).
     */
    struct array_local_type {
      /**
       * The array element type.
       */
      local_var_type element_type_;

      /**
       * The length of this array.
       */
      expression array_len_;

      /**
       * Construct an array local var type with default values.
       */
      array_local_type();

      /**
       * Construct a local array type with the specified element type
       * and array length.
       *
       * @param el_type element type 
       * @param len array length
       */
      array_local_type(const local_var_type& el_type,
                       const expression& len);
      
      /**
       * Returns type of elements stored in innermost array.
       */
      local_var_type contains() const;

      /**
       * Returns number of array dimensions.
       */
      size_t array_dims() const;

      /**
       * Returns the length of this array.
       */
      expression array_len() const;
    };
  }
}
#endif
