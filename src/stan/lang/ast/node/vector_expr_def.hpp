#ifndef STAN_LANG_AST_NODE_VECTOR_EXPR_DEF_HPP
#define STAN_LANG_AST_NODE_VECTOR_EXPR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    vector_expr::vector_expr() : args_(), N_() {  }

    vector_expr::vector_expr(const std::vector<expression>& args,
                             const expression& N) 
      : args_(args), N_(N){ }

    vector_expr& vector_expr::operator=(const vector_expr& ve) {
      args_ = ve.args_;
      N_ = ve.N_;
      return *this;
    }

  }
}
#endif
