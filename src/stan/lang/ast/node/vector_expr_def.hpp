#ifndef STAN_LANG_AST_NODE_VECTOR_EXPR_DEF_HPP
#define STAN_LANG_AST_NODE_VECTOR_EXPR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    vector_expr::vector_expr() : args_(), type_(VECTOR_T, 1U) {  }

    vector_expr::vector_expr(const std::vector<expression>& args)
      : args_(args), type_() { }

    vector_expr& vector_expr::operator=(const vector_expr& al) {
      args_ = al.args_;
      type_ = al.type_;
      return *this;
    }

  }
}
#endif
