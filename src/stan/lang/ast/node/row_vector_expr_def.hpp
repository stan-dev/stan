#ifndef STAN_LANG_AST_NODE_ROW_VECTOR_EXPR_DEF_HPP
#define STAN_LANG_AST_NODE_ROW_VECTOR_EXPR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    row_vector_expr::row_vector_expr() : args_(), N_() { }

    row_vector_expr::row_vector_expr(const std::vector<expression>& args,
                             const expression& N) 
      : args_(args), N_(N) { }

    row_vector_expr& row_vector_expr::operator=(const row_vector_expr& ve) {
      args_ = ve.args_;
      N_ = ve.N_;
      return *this;
    }

  }
}
#endif
