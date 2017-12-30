#ifndef STAN_LANG_AST_BLOCK_ARRAY_TYPE_HPP
#define STAN_LANG_AST_BLOCK_ARRAY_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>

namespace stan {
  namespace lang {

    struct block_var_type;
    
    /** 
     * Block array type for Stan variables and expressions (recursive).
     */
    struct block_array_type {
      /**
       * The array element type.
       */
      block_var_type element_type_;

      /**
       * The length of this array.
       */
      expression array_len_;

      /**
       * Construct an array block var type with default values.
       */
      block_array_type();

      /**
       * Construct a block array type with the specified element type
       * and array length.
       * Length should be int expression - constructor doesn't check.
       *
       * @param el_type element type 
       * @param len array length
       */
      block_array_type(const block_var_type& el_type,
                       const expression& len);

      /**
       * Returns type of elements stored in innermost array.
       */
      block_var_type contains() const;

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
