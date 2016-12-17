#ifndef STAN_LANG_AST_NODE_ARRAY_EXPR_DEF_HPP
#define STAN_LANG_AST_NODE_ARRAY_EXPR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    array_expr::array_expr() : args_(), type_(DOUBLE_T, 1U) {  }

    array_expr::array_expr(const std::vector<expression>& args)
      : args_(args), type_() { }

    array_expr& array_expr::operator=(const array_expr& al) {
      args_ = al.args_;
      type_ = al.type_;
      return *this;
    }

  }
}
#endif
