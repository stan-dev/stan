#ifndef STAN_LANG_AST_NODE_ROW_VECTOR_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_ROW_VECTOR_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    row_vector_var_decl::row_vector_var_decl() : base_var_decl(ROW_VECTOR_T) { }

    row_vector_var_decl::row_vector_var_decl(range const& range,
                                         expression const& N,
                                         std::string const& name,
                                         std::vector<expression> const& dims,
                                         expression const& def)
      : base_var_decl(name, dims, ROW_VECTOR_T, def), range_(range), N_(N) { }

  }
}
#endif
