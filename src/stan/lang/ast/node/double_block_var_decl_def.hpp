#ifndef STAN_LANG_AST_NODE_DOUBLE_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_DOUBLE_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    double_block_var_decl::double_block_var_decl() { }

    double_block_var_decl::double_block_var_decl(const std::string& name,
                                                 const range& bounds,
                                                 const expression& def)
      : var_decl(name, bare_expr_type(double_type()), def),
        type_(double_block_type(bounds)) { }
  }
}
#endif
