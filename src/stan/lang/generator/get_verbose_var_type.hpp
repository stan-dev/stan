#ifndef STAN_LANG_GENERATOR_GET_VERBOSE_VAR_TYPE_HPP
#define STAN_LANG_GENERATOR_GET_VERBOSE_VAR_TYPE_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {
    /**
     * Return cpp type used for variable declarations.
     *
     * @param[in] bare_expr_type bare_type
     */
    std::string
    get_verbose_var_type(const bare_expr_type bare_type) {

      //TODO:mitzi  why not use cpp_typename instead?

      bare_expr_type bt(bare_type);
      if (bt.is_array_type())
        bt = bt.array_contains();
      
      if (bt.is_matrix_type()) {
        return "Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, Eigen::Dynamic>";
      } else if (bt.is_row_vector_type()) {
        return "Eigen::Matrix<local_scalar_t__, 1, Eigen::Dynamic>";
      } else if (bt.is_vector_type()) {
        return "Eigen::Matrix<local_scalar_t__, Eigen::Dynamic, 1>";
      } else if (bt.is_double_type()) {
        return "local_scalar_t__";   // gets typedef'd in ctor, log_prob methods
      } else if (bt.is_int_type()) {
        return "int";
      }
      return "ill_formed";
    }
  }
}
#endif
