#ifndef STAN_LANG_AST_NODE_UNIT_VECTOR_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_UNIT_VECTOR_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    unit_vector_var_decl::unit_vector_var_decl() : base_var_decl(VECTOR_T) { }

    unit_vector_var_decl::unit_vector_var_decl(expression const& K,
                                       std::string const& name,
                                       std::vector<expression> const& dims)
      : base_var_decl(name, dims, VECTOR_T),  K_(K) { }

  }
}
#endif
