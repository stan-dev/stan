#ifndef STAN_LANG_AST_NODE_FUN_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_FUN_VAR_DECL_HPP

#include <boost/variant/recursive_variant.hpp>
#include <string>

namespace stan {
  namespace lang {

    struct nil;
    struct array_fun_var_decl;
    struct double_fun_var_decl;
    struct int_fun_var_decl;
    struct matrix_fun_var_decl;
    struct row_vector_fun_var_decl;
    struct vector_fun_var_decl;

    /**
     * The variant structure to hold a variable declaration.
     */
    struct fun_var_decl {
      /**
       * The variant type for a variable declaration.
       */
      typedef boost::variant<boost::recursive_wrapper<nil>,
                             boost::recursive_wrapper<array_fun_var_decl>,
                             boost::recursive_wrapper<int_fun_var_decl>,
                             boost::recursive_wrapper<double_fun_var_decl>,
                             boost::recursive_wrapper<vector_fun_var_decl>,
                             boost::recursive_wrapper<row_vector_fun_var_decl>,
                             boost::recursive_wrapper<matrix_fun_var_decl> >
      fun_var_decl_t;

      /**
       * The fun variable decl type held by this wrapper.
       */
      fun_var_decl_t var_decl_;

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
      fun_var_decl();

      /**
       * Construct a variable declaration.
       *
       * @param x variable declaration
       */
      fun_var_decl(const fun_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified variant
       * type holding a declaration.
       *
       * @param x variable declaration raw variant type
       */
      fun_var_decl(const fun_var_decl_t& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.
       *
       * @param x variable declaration
       */
      fun_var_decl(const nil& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      fun_var_decl(const int_fun_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      fun_var_decl(const double_fun_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      fun_var_decl(const vector_fun_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      fun_var_decl(const row_vector_fun_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      fun_var_decl(const matrix_fun_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param x variable declaration
       */
      fun_var_decl(const array_fun_var_decl& x);  // NOLINT(runtime/explicit)

      /**
       * Return the variable declaration's bare expr type.
       *
       * @return the bare expr type
       */
      bare_expr_type bare_type() const;

      /**
       * Return the variable declaration's fun_var_type.
       *
       * @return the fun_var_type
       */
      fun_var_type fun_var_type() const;

      /**
       * Return true if fun_var_decl has `data` qualifier
       *
       * @return bool
       */
      bool is_data() const;

      /**
       * Return the declaration's variable name.
       *
       * @return name of variable
       */
      std::string name() const;
    };

    /**
     * Stream a user-readable version of the fun_var_decl to the
     * specified output stream, returning the speicifed argument
     * output stream to allow chaining.
     *
     * @param o output stream
     * @param fvar fun_var_decl
     * @return argument output stream
     */
    std::ostream& operator<<(std::ostream& o,
                             const fun_var_decl& fv_decl);
  }
}
#endif
