#ifndef STAN_LANG_AST_SIGS_FUNCTION_SIGNATURE_T_HPP
#define STAN_LANG_AST_SIGS_FUNCTION_SIGNATURE_T_HPP

#include <stan/lang/ast/type/bare_expr_type.hpp>
#include <stan/lang/ast/type/fun_var_type.hpp>
#include <utility>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * The type of a function signature, mapping a vector of
     * argument expression types to a result expression type.
     */
    typedef std::pair<bare_expr_type, std::vector<fun_var_type> >
    function_signature_t;

  }
}
#endif
