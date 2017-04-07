#ifndef STAN_LANG_AST_NODE_DOUBLE_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_DOUBLE_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    double_var_decl::double_var_decl() : base_var_decl(DOUBLE_T) { }

    double_var_decl::double_var_decl(const range& range,
                                     const std::string& name,
                                     const std::vector<expression>& dims,
                                     const expression& def)
      : base_var_decl(name, dims, DOUBLE_T, def), range_(range) { }

  }
}
#endif
