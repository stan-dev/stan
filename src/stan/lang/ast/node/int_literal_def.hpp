#ifndef STAN_LANG_AST_NODE_INT_LITERAL_DEF_HPP
#define STAN_LANG_AST_NODE_INT_LITERAL_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {



    int_literal::int_literal() : type_(INT_T) { }

    int_literal::int_literal(int val) : val_(val), type_(INT_T) { }

    int_literal::int_literal(const int_literal& il)
      : val_(il.val_), type_(il.type_) { }

    int_literal& int_literal::operator=(const int_literal& il) {
      val_ = il.val_;
      type_ = il.type_;
      return *this;
    }

  }
}
#endif
