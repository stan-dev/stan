#ifndef STAN_LANG_ASST_NODE_VAR_DECL_HPP
#define STAN_LANG_ASST_NODE_VAR_DECL_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_corr_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_factor_var_decl.hpp>
#include <stan/lang/ast/node/corr_matrix_var_decl.hpp>
#include <stan/lang/ast/node/cov_matrix_var_decl.hpp>
#include <stan/lang/ast/node/double_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/int_var_decl.hpp>
#include <stan/lang/ast/node/matrix_var_decl.hpp>
#include <stan/lang/ast/node/nil.hpp>
#include <stan/lang/ast/node/ordered_var_decl.hpp>
#include <stan/lang/ast/node/positive_ordered_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_var_decl.hpp>
#include <stan/lang/ast/node/simplex_var_decl.hpp>
#include <stan/lang/ast/node/unit_vector_var_decl.hpp>
#include <stan/lang/ast/node/vector_var_decl.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * The variant structure to hold a variable declaration.
     */
    struct var_decl {
      /**
       * The variant type for a variable declaration.
       */
      typedef boost::variant<boost::recursive_wrapper<nil>,
                         boost::recursive_wrapper<int_var_decl>,
                         boost::recursive_wrapper<double_var_decl>,
                         boost::recursive_wrapper<vector_var_decl>,
                         boost::recursive_wrapper<row_vector_var_decl>,
                         boost::recursive_wrapper<matrix_var_decl>,
                         boost::recursive_wrapper<simplex_var_decl>,
                         boost::recursive_wrapper<unit_vector_var_decl>,
                         boost::recursive_wrapper<ordered_var_decl>,
                         boost::recursive_wrapper<positive_ordered_var_decl>,
                         boost::recursive_wrapper<cholesky_factor_var_decl>,
                         boost::recursive_wrapper<cholesky_corr_var_decl>,
                         boost::recursive_wrapper<cov_matrix_var_decl>,
                         boost::recursive_wrapper<corr_matrix_var_decl> >
      var_decl_t;

      /**
       * Construct a default variable declaration.
       */
      var_decl();

      /**
       * Construct a variable declaration with the specified variant
       * type holding a declaration.
       *
       * @param decl variable declaration raw variant type holding a
       * basic declaration
       */
      var_decl(const var_decl_t& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.
       *
       * @param decl variable declaration
       */
      var_decl(const nil& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const int_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const double_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const vector_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const row_vector_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const matrix_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const simplex_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const unit_vector_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const ordered_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const positive_ordered_var_decl& decl);  // NOLINT

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const cholesky_factor_var_decl& decl);  // NOLINT

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const cholesky_corr_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const cov_matrix_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Construct a variable declaration with the specified
       * basic declaration.  
       *
       * @param decl variable declaration
       */
      var_decl(const corr_matrix_var_decl& decl);  // NOLINT(runtime/explicit)

      /**
       * Return the declaration's variable name.
       *
       * @return name of variable
       */
      std::string name() const;

      /**
       * Return the base declaration.
       *
       * @return base variable declaration
       */
      base_var_decl base_decl() const;

      /**
       * Return the sequence of array dimension sizes. 
       *
       * @return sequence of dimension sizes
       */
      std::vector<expression> dims() const;

      /**
       * Return true if this declaration also contains a definition. 
       *
       * @return true if there is a definition
       */
      bool has_def() const;

      /**
       * Return the definition included in this declaration.
       *
       * @return the variable definition
       */
      expression def() const;

      /**
       * The variable declaration variant type.
       */
      var_decl_t decl_;
    };

  }
}
#endif
