#ifndef STAN_LANG_AST_NODE_MATRIX_EXPR_DEF_HPP
#define STAN_LANG_AST_NODE_MATRIX_EXPR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    matrix_expr::matrix_expr() : args_() { }

    matrix_expr::matrix_expr(const std::vector<expression>& args)
      : args_(args) { }

    matrix_expr& matrix_expr::operator=(const matrix_expr& me) {
      args_ = me.args_;
      return *this;
    }

  }
}
#endif
