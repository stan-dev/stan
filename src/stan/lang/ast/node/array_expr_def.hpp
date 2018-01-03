#ifndef STAN_LANG_AST_NODE_ARRAY_EXPR_DEF_HPP
#define STAN_LANG_AST_NODE_ARRAY_EXPR_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    array_expr::array_expr() : args_(), type_(), has_var_(false),
                               array_expr_scope_() { }

    array_expr::array_expr(const std::vector<expression>& args)
      : args_(args), type_(bare_array_type(args.at(0).bare_type())),
        has_var_(false), array_expr_scope_() { }

  }
}
#endif
