#ifndef STAN_LANG_AST_FUN_WRITE_BASE_EXPR_TYPE_DEF_HPP
#define STAN_LANG_AST_FUN_WRITE_BASE_EXPR_TYPE_DEF_HPP

#include <stan/lang/ast/fun/write_base_expr_type.hpp>

namespace stan {
  namespace lang {

    std::ostream& write_base_expr_type(std::ostream& o, base_expr_type type) {
      if (type.is_int_type())
        o << "int";
      else if (type.is_double_type())
        o << "real";
      else if (type.is_vector_type())
        o << "vector";
      else if (type.is_row_vector_type())
        o << "row vector";
      else if (type.is_matrix_type())
        o << "matrix";
      else if (type.is_ill_formed_type())
        o << "ill formed";
      else if (type.is_void_type())
        o << "void";
      else
        o << "UNKNOWN";

      return o;
    }
  }
}

#endif
