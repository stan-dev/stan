#ifndef STAN_LANG_GENERATOR_GENERATE_REAL_VAR_TYPE_HPP
#define STAN_LANG_GENERATOR_GENERATE_REAL_VAR_TYPE_HPP

#include <stan/lang/ast.hpp>
#include <ostream>

namespace stan {
  namespace lang {

   /**
     * Generate correct C++ type for expressions which contain a Stan
     * <code>real</code> variable according to context in which
     * expression is used and expression contents.
     *
     * @param[in] vo expression origin block
     * @param[in] has_var  does expression contains a variable?
     * @param[in] is_var_context true when in auto-diff context
     * @param[in,out] o generated typename
     */
    void generate_real_var_type(const var_origin& vo,
                                bool has_var,
                                bool is_var_context,
                                std::ostream& o) {
      if (is_fun_origin(vo))
        o << "fun_scalar_t__";
      else if (is_var_context && has_var)
        o << "T__";
      else
        o << "double";
    }

  }
}
#endif
