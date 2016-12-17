#ifndef STAN_LANG_AST_FUN_WRITE_BASE_EXPR_TYPE_DEF_HPP
#define STAN_LANG_AST_FUN_WRITE_BASE_EXPR_TYPE_DEF_HPP

#include <stan/lang/ast/fun/write_base_expr_type.hpp>

namespace stan {
  namespace lang {

    std::ostream& write_base_expr_type(std::ostream& o, base_expr_type type) {
      switch (type) {
      case INT_T :
        o << "int";
        break;
      case DOUBLE_T :
        o << "real";
        break;
      case VECTOR_T :
        o << "vector";
        break;
      case ROW_VECTOR_T :
        o << "row vector";
        break;
      case MATRIX_T :
        o << "matrix";
        break;
      case ILL_FORMED_T :
        o << "ill formed";
        break;
      case VOID_T :
        o << "void";
        break;
      default:
        o << "UNKNOWN";
      }
      return o;
    }
  }
}

#endif
