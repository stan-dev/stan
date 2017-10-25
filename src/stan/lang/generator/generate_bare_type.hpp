#ifndef STAN_LANG_GENERATOR_GENERATE_BARE_TYPE_HPP
#define STAN_LANG_GENERATOR_GENERATE_BARE_TYPE_HPP

#include <stan/lang/ast.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {


    /**
     * Generate the basic type for the specified expression type
     * without dimensions, using the specified scalar type string,
     * writing to the specified stream.
     *
     * @param[in] t expression type
     * @param[in] scalar_t_name name of scalar type for double values
     * and containers
     * @param[in] o stream for generating
     */
    void generate_bare_type(const expr_type& t,
                            const std::string& scalar_t_name,
                            std::ostream& o) {
      for (size_t d = 0; d < t.num_dims_; ++d)
        o << "std::vector<";
      bool is_template_type = false;
      if (t.base_type_.is_int_type()) {
        o << "int";
        is_template_type = false;
      } else if (t.base_type_.is_double_type()) {
        o << scalar_t_name;
        is_template_type = false;
      } else if (t.base_type_.is_vector_type()) {
        o << "Eigen::Matrix<"
          << scalar_t_name
          << ", Eigen::Dynamic,1>";
        is_template_type = true;
      } else if (t.base_type_.is_row_vector_type()) {
        o << "Eigen::Matrix<"
          << scalar_t_name
          << ", 1,Eigen::Dynamic>";
        is_template_type = true;
      } else if (t.base_type_.is_matrix_type()) {
        o << "Eigen::Matrix<"
          << scalar_t_name
          << ", Eigen::Dynamic,Eigen::Dynamic>";
        is_template_type = true;
      } else if (t.base_type_.is_void_type()) {
        o << "void";
      } else {
        o << "UNKNOWN TYPE";
      }
      for (size_t d = 0; d < t.num_dims_; ++d) {
        if (d > 0 || is_template_type)
          o << " ";
        o << ">";
      }
    }

  }
}
#endif
