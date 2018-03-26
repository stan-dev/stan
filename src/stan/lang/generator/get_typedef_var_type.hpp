#ifndef STAN_LANG_GENERATOR_GET_TYPEDEF_VAR_TYPE_HPP
#define STAN_LANG_GENERATOR_GET_TYPEDEF_VAR_TYPE_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {
    /**
     * Return cpp type name or typedef used for bare_expr_type.
     * NOTE: (26/02/2018)
     *   typdefs come from `lib/stan_math/prim/mat/fun/typedefs.hpp`
     *   vectors and matrices of double (appropriate for data, not params)
     *
     * @param[in] bare_type bare_type
     */
    std::string
    get_typedef_var_type(const bare_expr_type bare_type) {
      bare_expr_type bt(bare_type);
      if (bt.is_array_type())
        bt = bt.array_contains();

      if (bt.is_matrix_type()) {
        return "matrix_d";
      } else if (bt.is_row_vector_type()) {
        return "row_vector_d";
      } else if (bt.is_vector_type()) {
        return "vector_d";
      } else if (bt.is_double_type()) {
        return "double";
      } else if (bt.is_int_type()) {
        return "int";
      }
      return "ill_formed";
    }
  }
}
#endif
