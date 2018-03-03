#ifndef STAN_LANG_GENERATOR_GENERATE_VAR_DECL_HPP
#define STAN_LANG_GENERATOR_GENERATE_VAR_DECL_HPP

#include <stan/lang/ast.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {


    /**
     * Generate a variable declaration using the specified name,
     * type, and specified scalar type string, and writing to
     * the specified stream.
     * NOTE: declaration is not terminated.
     *
     * Scalar type string is `local_scalar_t__` in log_prob method,
     * `double` elsewhere.
     *
     * @param[in] var_name variable name
     * @param[in] t expression type
     * @param[in] scalar_t_name name of scalar type for double values
     * @param[in] o stream for generating
     */
    void generate_var_decl(const std::string& var_name,
                           const bare_expr_type& t,
                           const std::string& scalar_t_name,
                           std::ostream& o) {
      generate_bare_type(t, scalar_t_name, o);
      o << " " << var_name;
    }

  }
}
#endif
