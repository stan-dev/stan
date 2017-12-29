#ifndef STAN_LANG_AST_LOCAL_VAR_TYPE_HPP
#define STAN_LANG_AST_LOCAL_VAR_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <string>

namespace stan {
  namespace lang {

    /** 
     * Local variable types have sized container types.
     */
    
    struct array_local_type;
    struct double_type;
    struct ill_formed_type;
    struct int_type;
    struct matrix_local_type;
    struct row_vector_local_type;
    struct vector_local_type;

    struct local_var_type {
      /**
       * Recursive wrapper for local variable types.
       */
      typedef boost::variant<
        boost::recursive_wrapper<ill_formed_type>,
        boost::recursive_wrapper<double_type>,
        boost::recursive_wrapper<int_type>,
        boost::recursive_wrapper<matrix_local_type>,
        boost::recursive_wrapper<row_vector_local_type>,
        boost::recursive_wrapper<vector_local_type>,
        boost::recursive_wrapper<array_local_type> >
      local_t;

      /**
       * The local variable type held by this wrapper.
       */
      local_t var_type_;

      /**
       * Construct a bare var type with default values.
       */
      local_var_type();

      /**
       * Construct a local var type 
       *
       * @param type local variable type raw variant type.
       */
      local_var_type(const local_var_type& type);  // NOLINT(runtime/explicit)

      /**
       * Construct a local var type with the specified type.
       *
       * @param type local variable type
       */      
      local_var_type(const ill_formed_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a local var type with the specified type.
       *
       * @param type local variable type
       */      
      local_var_type(const double_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a local var type with the specified type.
       *
       * @param type local variable type
       */      

      local_var_type(const int_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a local var type with the specified type.
       *
       * @param type local variable type
       */      
      local_var_type(const matrix_local_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a local var type with the specified type.
       *
       * @param type local variable type
       */      
      local_var_type(const row_vector_local_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a local var type with the specified type.
       *
       * @param type local variable type
       */      
      local_var_type(const vector_local_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a local var type with the specified type.
       *
       * @param type local variable type
       */      
      local_var_type(const array_local_type& type); // NOLINT(runtime/explicit)

      /**
       * Construct a local var type with the specified type.
       *
       * @param type local variable type
       */
      local_var_type(const local_t& var_type_);  // NOLINT(runtime/explicit)

      
      /**
       * Returns true if `var_type_` is `array_local_type`, false otherwise.
       */
      bool is_array_var_type() const;

      /**
       * Returns array element type if `var_type_` is `array_local_type`,
       * ill_formed_type otherwise.  (Call `is_array_var_type()` first.)
       */
      local_var_type get_array_el_type() const;

      /**
       * Returns total number of dimensions for container type.
       * Returns 0 for scalar types.
       */
      int num_dims() const;

      /**
       * Returns vector of sizes for each dimension, nil if unsized.
       */
      std::vector<expression> size() const;
    };
    
  }
}
#endif
