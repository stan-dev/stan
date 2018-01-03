#ifndef STAN_LANG_AST_NODE_BLOCK_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_BLOCK_VAR_DECL_HPP

#include <stan/lang/ast/type/block_var_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    struct nil;
    struct array_block_var_decl;
    struct cholesky_corr_block_var_decl;
    struct cholesky_factor_block_var_decl;
    struct corr_matrix_block_var_decl;
    struct cov_matrix_block_var_decl;
    struct double_block_var_decl;
    struct int_block_var_decl;
    struct matrix_block_var_decl;
    struct ordered_block_var_decl;
    struct positive_ordered_block_var_decl;
    struct row_vector_block_var_decl;
    struct simplex_block_var_decl;
    struct unit_vector_block_var_decl;
    struct vector_block_var_decl;

    /**
     * The variant structure to hold a variable declaration.
     */
    struct block_var_decl {

      /**
       * The variant type for a variable declaration.
       */
      typedef boost::variant<boost::recursive_wrapper<nil>,
                             boost::recursive_wrapper<array_block_var_decl>,
                             boost::recursive_wrapper<int_block_var_decl>,
                             boost::recursive_wrapper<double_block_var_decl>,
                             boost::recursive_wrapper<vector_block_var_decl>,
                             boost::recursive_wrapper<row_vector_block_var_decl>,
                             boost::recursive_wrapper<matrix_block_var_decl>,
                             boost::recursive_wrapper<simplex_block_var_decl>,
                             boost::recursive_wrapper<unit_vector_block_var_decl>,
                             boost::recursive_wrapper<ordered_block_var_decl>,
                             boost::recursive_wrapper<positive_ordered_block_var_decl>,
                             boost::recursive_wrapper<cholesky_factor_block_var_decl>,
                             boost::recursive_wrapper<cholesky_corr_block_var_decl>,
                             boost::recursive_wrapper<cov_matrix_block_var_decl>,
                             boost::recursive_wrapper<corr_matrix_block_var_decl> >
      block_var_decl_t;

      /**
       * The block variable decl type held by this wrapper.
       */
      block_var_decl_t var_decl_;

      /**
       * The line in the source code where the declaration begins.
       */
      std::size_t begin_line_;

      /**
       * The line in the source code where the declaration ends.
       */
      std::size_t end_line_;

      /**
       * Construct a default variable declaration.
       */
      block_var_decl();

      /**
       * Construct a variable declaration.
       *
       * @param x variable declaration
       * basic declaration
       */
      block_var_decl(const block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified variant
       * type holding a declaration.
       *
       * @param x variable declaration raw variant type holding a
       * basic declaration
       */
      block_var_decl(const block_var_decl_t& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.
       *
       * @param x variable declaration
       */
      block_var_decl(const nil& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const int_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const double_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const vector_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const row_vector_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const matrix_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const simplex_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const unit_vector_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const ordered_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const positive_ordered_block_var_decl& x);  // NOLINT

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const cholesky_factor_block_var_decl& x);  // NOLINT

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const cholesky_corr_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const cov_matrix_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const corr_matrix_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      block_var_decl(const array_block_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Return the variable declaration's bare expr type.
       *
       * @return the bare expr type
       */
      bare_expr_type bare_type() const;

      /**
       * Return the variable declaration's definition.
       *
       * @return expression definition for this variable
       */
      expression def() const;

      /**
       * Return true if variable declaration contains a definition.
       *
       * @return bool indicating has or doesn't have definition
       */
      bool has_def() const;

      /**
       * Return the variable declaration's name.
       *
       * @return name of variable
       */
      std::string name() const;

      /**
       * Return the variable declaration's block_var_type
       *
       * @return the variable expression's block_var_type
       */
      block_var_type type() const;

      /**
       * Return the var_decl for this variable declaration
       *
       * @return var_decl
      */
      var_decl var_decl() const;
    };
  }
}
#endif
