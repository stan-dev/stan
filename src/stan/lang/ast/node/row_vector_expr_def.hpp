#ifndef STAN_LANG_AST_NODE_ROW_VECTOR_EXPR_DEF_HPP
#define STAN_LANG_AST_NODE_ROW_VECTOR_EXPR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    row_vector_expr::row_vector_expr() : args_(), type_(ROW_VECTOR_T, 1U) {  }

    row_vector_expr::row_vector_expr(const std::vector<expression>& args)
      : args_(args), type_() { }

    row_vector_expr& row_vector_expr::operator=(const row_vector_expr& al) {
      args_ = al.args_;
      type_ = al.type_;
      return *this;
    }

  }
}
#endif
