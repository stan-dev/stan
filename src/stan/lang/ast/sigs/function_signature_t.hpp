#ifndef STAN_LANG_AST_SIGS_FUNCTION_SIGNATURE_T_HPP
#define STAN_LANG_AST_SIGS_FUNCTION_SIGNATURE_T_HPP

#include <stan/lang/ast/sigs/function_arg_type.hpp>
#include <utility>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * The type of a function signature, mapping a vector of
     * argument expression types to a result expression type.
     */
    typedef std::pair<expr_type, std::vector<function_arg_type> > function_signature_t;

  }
}
#endif
