#ifndef STAN_LANG_AST_NODE_MATRIX_EXPR_DEF_HPP
#define STAN_LANG_AST_NODE_MATRIX_EXPR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    matrix_expr::matrix_expr() : args_(), M_(), N_() { }

    matrix_expr::matrix_expr(const std::vector<expression>& args,
                             const expression& M, const expression& N)
      : args_(args), M_(M), N_(N) { }

    matrix_expr& matrix_expr::operator=(const matrix_expr& me) {
      args_ = me.args_;
      M_ = me.M_;
      N_ = me.N_;
      return *this;
    }

  }
}
#endif
