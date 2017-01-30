#ifndef STAN_LANG_AST_NODE_GENERALODEMODEL_CONTROL_DEF_HPP
#define STAN_LANG_AST_NODE_GENERALODEMODEL_CONTROL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    generalOdeModel_control::generalOdeModel_control() { }

    generalOdeModel_control::generalOdeModel_control(
                           const std::string& integration_function_name,
                           const std::string& system_function_name,
                           const expression& nCmt,
                           const expression& time,
                           const expression& amt,
                           const expression& rate,
                           const expression& ii,
                           const expression& evid,
                           const expression& cmt,
                           const expression& addl,
                           const expression& ss,
                           const expression& pMatrix,
                           const expression& biovar,
                           const expression& tlag,
                           const expression& rel_tol, const expression& abs_tol,
                           const expression& max_num_steps)
      : integration_function_name_(integration_function_name),
        system_function_name_(system_function_name),
        nCmt_(nCmt), time_(time), amt_(amt), rate_(rate), ii_(ii),
        evid_(evid), cmt_(cmt), addl_(addl), ss_(ss), pMatrix_(pMatrix),
        biovar_(biovar), tlag_(tlag), rel_tol_(rel_tol), abs_tol_(abs_tol),
        max_num_steps_(max_num_steps) {
    }

  }
}
#endif
