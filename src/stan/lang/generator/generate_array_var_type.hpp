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
     * @param[in,out] o generated typename
     */
    void generate_array_var_type(const base_expr_type& base_type,
                                 const std::string& real_var_type,
                                 std::ostream& o) {
      switch (base_type) {
      case INT_T :
        o << "int";
        break;
      case DOUBLE_T :
        o << real_var_type;
        break;
      case VECTOR_T :
        o << "Eigen::Matrix<" << real_var_type << ",Eigen::Dynamic,1> ";
        break;
      case ROW_VECTOR_T :
        o << "Eigen::Matrix<" << real_var_type << ",1,Eigen::Dynamic> ";
        break;
      case MATRIX_T :
        o << "Eigen::Matrix<" << real_var_type
          << ",Eigen::Dynamic,Eigen::Dynamic> ";
        break;
      }
    }



  }
}
#endif
