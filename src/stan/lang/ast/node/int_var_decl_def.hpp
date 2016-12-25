#ifndef STAN_LANG_AST_NODE_INT_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_INT_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    int_var_decl::int_var_decl() : base_var_decl(INT_T) { }

    int_var_decl::int_var_decl(const range& range, const std::string& name,
                               const std::vector<expression>& dims,
                               const expression& def)
      : base_var_decl(name, dims, INT_T, def), range_(range) { }

  }
}
#endif
