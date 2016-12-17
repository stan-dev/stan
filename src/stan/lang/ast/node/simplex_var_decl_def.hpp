#ifndef STAN_LANG_AST_NODE_SIMPLEX_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_SIMPLEX_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    simplex_var_decl::simplex_var_decl() : base_var_decl(VECTOR_T) { }

    simplex_var_decl::simplex_var_decl(expression const& K,
                                       std::string const& name,
                                       std::vector<expression> const& dims)
      : base_var_decl(name, dims, VECTOR_T),  K_(K) { }

  }
}
#endif
