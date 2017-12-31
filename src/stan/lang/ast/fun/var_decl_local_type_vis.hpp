#ifndef STAN_LANG_AST_FUN_VAR_DECL_LOCAL_TYPE_VIS_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_LOCAL_TYPE_VIS_HPP

#include <stan/lang/ast/nil.hpp>
#include <stan/lang/ast/type/local_var_type.hpp>
#include <stan/lang/ast/node/array_local_var_decl.hpp>
#include <stan/lang/ast/node/double_local_var_decl.hpp>
#include <stan/lang/ast/node/int_local_var_decl.hpp>
#include <stan/lang/ast/node/matrix_local_var_decl.hpp>
#include <stan/lang/ast/node/row_vector_local_var_decl.hpp>
#include <stan/lang/ast/node/vector_local_var_decl.hpp>
#include <boost/variant/static_visitor.hpp>


namespace stan {
  namespace lang {

    /**
     * A visitor to get local_var_type from local_var_decls.
     */
    struct var_decl_local_type_vis
      : public boost::static_visitor<local_var_type> {
      /**
       * Construct a var_decl_type visitor.
       */
      var_decl_local_type_vis();

      /**
       * Return the ill-formed type
       *
       * @param x variable declaration
       * @return ill_formed_type
       */
      local_var_type operator()(const nil& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      local_var_type operator()(const array_local_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      local_var_type operator()(const int_local_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      local_var_type operator()(const double_local_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      local_var_type operator()(const vector_local_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      local_var_type operator()(const row_vector_local_var_decl& x) const;

      /**
       * Return the type of the variable.
       *
       * @param x variable declaration
       * @return the type of the variable being declared
       */
      local_var_type operator()(const matrix_local_var_decl& x) const;
    };

  }
}
#endif
