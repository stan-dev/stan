#ifndef STAN_LANG_AST_NODE_VECTOR_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_VECTOR_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    vector_var_decl::vector_var_decl() : base_var_decl(VECTOR_T) { }

    vector_var_decl::vector_var_decl(range const& range, expression const& M,
                                     std::string const& name,
                                     std::vector<expression> const& dims,
                                     expression const& def)
      : base_var_decl(name, dims, VECTOR_T, def), range_(range), M_(M) {  }

  }
}
#endif
