#ifndef STAN_LANG_GENERATOR_GENERATE_BARE_TYPE_HPP
#define STAN_LANG_GENERATOR_GENERATE_BARE_TYPE_HPP

#include <stan/lang/ast.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {


    /**
     * Generate the basic type for the specified expression type
     * using the specified scalar type string and writing to
     * the specified stream.
     *
     * Scalar type string is `local_scalar_t__` in log_prob method,
     * `double` elsewhere.
     *
     * @param[in] t expression type
     * @param[in] scalar_t_name name of scalar type for double values
     * @param[in] o stream for generating
     */
    void generate_bare_type(const bare_expr_type& t,
                            const std::string& scalar_t_name,
                            std::ostream& o) {
      // unfold array type to get array element info
      int num_dims = t.array_dims();
      bare_expr_type bt(t);
      if (bt.is_array_type())
          bt = bt.array_contains();

      for (int d = 0; d < num_dims; ++d)
        o << "std::vector<";
      bool is_template_type = false;
      if (bt.is_int_type()) {
        o << "int";
        is_template_type = false;
      } else if (bt.is_double_type()) {
        o << scalar_t_name;
        is_template_type = false;
      } else if (bt.is_vector_type()) {
        o << "Eigen::Matrix<"
          << scalar_t_name
          << ", Eigen::Dynamic,1>";
        is_template_type = true;
      } else if (bt.is_row_vector_type()) {
        o << "Eigen::Matrix<"
          << scalar_t_name
          << ", 1, Eigen::Dynamic>";
        is_template_type = true;
      } else if (bt.is_matrix_type()) {
        o << "Eigen::Matrix<"
          << scalar_t_name
          << ", Eigen::Dynamic, Eigen::Dynamic>";
        is_template_type = true;
      } else if (bt.is_void_type()) {
        o << "void";
      } else {
        o << "UNKNOWN TYPE";
      }
      for (int d = 0; d < num_dims; ++d) {
        if (d > 0 || is_template_type)
          o << " ";
        o << ">";
      }
    }

  }
}
#endif
