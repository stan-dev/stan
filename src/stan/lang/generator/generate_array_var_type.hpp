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
     * @param[in] base_type expression base type
     * @param[in] real_var_type context-dependent <code>real</code> type
     * @param[in] is_var_context true when in auto-diff context
     * @param[in,out] o generated typename
     */
    void generate_array_var_type(const base_expr_type& base_type,
                                 const std::string& real_var_type,
                                 bool is_var_context,
                                 std::ostream& o) {
      switch (base_type) {
      case INT_T :
        o << "int";
        break;
      case DOUBLE_T :
        o << real_var_type;
        break;
      case VECTOR_T :
        o << (is_var_context
              ? "Eigen::Matrix<T__,Eigen::Dynamic,1> "
              : "vector_d");
        break;
      case ROW_VECTOR_T :
        o << (is_var_context
              ? "Eigen::Matrix<T__,1,Eigen::Dynamic> "
              : "row_vector_d");
        break;
      case MATRIX_T :
        o << (is_var_context
              ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
              : "matrix_d");
        break;
      }
    }



  }
}
#endif
