#ifndef STAN_LANG_GENERATOR_GENERATE_ARRAY_VAR_TYPE_HPP
#define STAN_LANG_GENERATOR_GENERATE_ARRAY_VAR_TYPE_HPP

#include <stan/lang/ast.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate C++ type for array expressions according to context in
     * which expression appears.
     *
     * @param[in] bare_type expression bare type
     * @param[in] real_var_type context-dependent <code>real</code> type
     * @param[in,out] o generated typename
     */
    void generate_array_var_type(const bare_expr_type& bare_type,
                                 const std::string& real_var_type,
                                 std::ostream& o) {
      if (bare_type.is_int_type())
        o << "int";
      else if (bare_type.is_double_type())
        o << real_var_type;
      else if (bare_type.is_vector_type())
        o << "Eigen::Matrix<" << real_var_type << ", Eigen::Dynamic, 1> ";
      else if (bare_type.is_row_vector_type())
        o << "Eigen::Matrix<" << real_var_type << ", 1, Eigen::Dynamic> ";
      else if (bare_type.is_matrix_type())
        o << "Eigen::Matrix<" << real_var_type
          << ", Eigen::Dynamic, Eigen::Dynamic> ";
    }

  }
}
#endif
