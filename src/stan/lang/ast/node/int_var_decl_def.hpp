#ifndef STAN_LANG_AST_NODE_INT_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_INT_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    int_var_decl::int_var_decl() : base_var_decl(INT_T) { }

    int_var_decl::int_var_decl(range const& range, std::string const& name,
                               std::vector<expression> const& dims,
                               expression const& def)
      : base_var_decl(name, dims, INT_T, def), range_(range) { }

  }
}
#endif
