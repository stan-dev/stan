#ifndef STAN_LANG_AST_BLOCK_VAR_TYPE_HPP
#define STAN_LANG_AST_BLOCK_VAR_TYPE_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <string>

namespace stan {
  namespace lang {

    /** 
     * Block variable types
     */

    struct array_block_type;
    struct cholesky_corr_block_type;
    struct cholesky_factor_block_type;
    struct corr_matrix_block_type;
    struct cov_matrix_block_type;
    struct double_block_type;
    struct ill_formed_type;
    struct int_block_type;
    struct matrix_block_type;
    struct ordered_block_type;
    struct positive_ordered_block_type;
    struct row_vector_block_type;
    struct simplex_block_type;
    struct unit_vector_block_type;
    struct vector_block_type;
    
    struct block_var_type {
      /**
       * Recursive wrapper for block variable types.
       */
      typedef boost::variant<
        boost::recursive_wrapper<ill_formed_type>,
        boost::recursive_wrapper<cholesky_corr_block_type>,
        boost::recursive_wrapper<cholesky_factor_block_type>,
        boost::recursive_wrapper<corr_matrix_block_type>,
        boost::recursive_wrapper<cov_matrix_block_type>,
        boost::recursive_wrapper<double_block_type>,
        boost::recursive_wrapper<int_block_type>,
        boost::recursive_wrapper<matrix_block_type>,
        boost::recursive_wrapper<ordered_block_type>,
        boost::recursive_wrapper<positive_ordered_block_type>,
        boost::recursive_wrapper<row_vector_block_type>,
        boost::recursive_wrapper<simplex_block_type>,
        boost::recursive_wrapper<unit_vector_block_type>,
        boost::recursive_wrapper<vector_block_type>,
        boost::recursive_wrapper<array_block_type> >
      block_t;

      /**
       * The block variable type held by this wrapper.
       */
      block_t var_type_;

      /**
       * Construct a block var type with default values.
       */
      block_var_type();

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */
      block_var_type(const block_var_type& type);  // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const ill_formed_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const cholesky_corr_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const cholesky_factor_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const corr_matrix_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const cov_matrix_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const double_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const int_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const matrix_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const ordered_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const positive_ordered_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const row_vector_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const simplex_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const unit_vector_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const vector_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */      
      block_var_type(const array_block_type& x); // NOLINT(runtime/explicit)

      /**
       * Construct a block var type with the specified type.
       *
       * @param type block variable type
       */
      block_var_type(const block_t& var_type_);  // NOLINT(runtime/explicit)

      /**
       * Returns true if `var_type_` is `array_block_type`, false otherwise.
       */
      bool is_array_var_type() const;

      /**
       * Returns array element type if `var_type_` is `array_block_type`,
       * ill_formed_type otherwise.  (Call `is_array_var_type()` first.)
       */
      block_var_type get_array_el_type() const;

      /**
       * Returns total number of dimensions for container type.
       * Returns 0 for scalar types.
       */
      int num_dims() const;

      // /**
      //  * Returns vector of sizes for each dimension, nil if unsized.
      //  */
      // std::vector<expression> size() const;
    };
  }
}
#endif
