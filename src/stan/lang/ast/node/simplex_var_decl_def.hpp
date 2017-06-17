#ifndef STAN_LANG_AST_NODE_SIMPLEX_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_SIMPLEX_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    simplex_var_decl::simplex_var_decl() : base_var_decl(VECTOR_T) { }

    simplex_var_decl::simplex_var_decl(const expression& K,
                                       const std::string& name,
                                       const std::vector<expression>& dims,
                                       const expression& def)

      : base_var_decl(name, dims, VECTOR_T, def),  K_(K) { }

  }
}
#endif
