#ifndef STAN_LANG_AST_NODE_LOCAL_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_LOCAL_VAR_DECL_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    struct nil;
    struct array_local_var_decl;
    struct double_local_var_decl;
    struct int_local_var_decl;
    struct matrix_local_var_decl;
    struct row_vector_local_var_decl;
    struct vector_local_var_decl;
    
    /**
     * The variant structure to hold a variable declaration.
     */
    struct local_var_decl {

      /**
       * The variant type for a variable declaration.
       */
      typedef boost::variant<
        boost::recursive_wrapper<nil>,
        boost::recursive_wrapper<array_local_var_decl>,
        boost::recursive_wrapper<int_local_var_decl>,
        boost::recursive_wrapper<double_local_var_decl>,
        boost::recursive_wrapper<vector_local_var_decl>,
        boost::recursive_wrapper<row_vector_local_var_decl>,
        boost::recursive_wrapper<matrix_local_var_decl> >
      local_var_decl_t;

      /**
       * The local variable decl type held by this wrapper.
       */
      local_var_decl_t var_decl_;

      // do we need data qualifier for local vars?
      // /**
      //  * True if variable has "data" qualifier.
      //  */
      // bool is_data_;

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
      local_var_decl();

      /**
       * Construct a variable declaration with the specified variant
       * type holding a declaration.
       *
       * @param decl variable declaration raw variant type holding a
       * basic declaration
       */
      local_var_decl(const local_var_decl_t& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.
       *
       * @param decl variable declaration
       */
      local_var_decl(const nil& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      local_var_decl(const int_local_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      local_var_decl(const double_local_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      local_var_decl(const vector_local_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      local_var_decl(const row_vector_local_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      local_var_decl(const matrix_local_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      local_var_decl(const array_local_var_decl& x);  // NOLINT(runtime/explicit)

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
       * Return the variable declaration's local_var_type
       *
       * @return the variable expression's local_var_type
       */
      local_var_type type() const;
    };

  }
}
#endif
