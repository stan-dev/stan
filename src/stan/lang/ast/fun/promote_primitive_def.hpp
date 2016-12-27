#ifndef STAN_LANG_AST_FUN_PROMOTE_PRIMITIVE_DEF_HPP
#define STAN_LANG_AST_FUN_PROMOTE_PRIMITIVE_DEF_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/fun/promote_primitive.hpp>

namespace stan {
  namespace lang {

   expr_type promote_primitive(const expr_type& et) {
      if (!et.is_primitive())
        return expr_type();
      return et;
    }

    expr_type promote_primitive(const expr_type& et1,
                                const expr_type& et2) {
      if (!et1.is_primitive() || !et2.is_primitive())
        return expr_type();
      return et1.type() == DOUBLE_T ? et1 : et2;
    }

  }
}
#endif
