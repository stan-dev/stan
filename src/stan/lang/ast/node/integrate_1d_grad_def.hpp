#ifndef STAN_LANG_AST_NODE_INTEGRATE_1D_GRAD_DEF_HPP
#define STAN_LANG_AST_NODE_INTEGRATE_1D_GRAD_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    integrate_1d_grad::integrate_1d_grad() { }

    integrate_1d_grad::integrate_1d_grad(
                           const std::string& integration_function_name,
                           const std::string& system_function_1_name,
                           const std::string& system_function_2_name,
                           const expression& a,  const expression& b,
                           const expression& param)
      : integration_function_name_(integration_function_name),
        system_function_1_name_(system_function_1_name),
        system_function_2_name_(system_function_2_name),
        a_(a), b_(b), param_(param) {
    }

  }
}
#endif
