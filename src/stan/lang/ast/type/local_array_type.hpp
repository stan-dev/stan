#ifndef STAN_LANG_AST_LOCAL_ARRAY_TYPE_HPP
#define STAN_LANG_AST_LOCAL_ARRAY_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>

namespace stan {
  namespace lang {

    struct local_var_type;

    /** 
     * Local array type for Stan variables and expressions (recursive).
     */
    struct local_array_type {
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
      local_array_type();

      /**
       * Construct a local array type with the specified element type
       * and array length.
       * Length should be int expression - constructor doesn't check.
       *
       * @param el_type element type 
       * @param len array length
       */
      local_array_type(const local_var_type& el_type,
                       const expression& len);
      
      /**
       * Returns type of elements stored in innermost array.
       */
      local_var_type contains() const;

      /**
       * Returns number of array dimensions.
       */
      int dims() const;

      /**
       * Returns the length of this array.
       */
      expression array_len() const;
    };
  }
}
#endif
