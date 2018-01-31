#ifndef STAN_LANG_AST_NODE_VECTOR_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_VECTOR_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    vector_block_var_decl::vector_block_var_decl() { }

    vector_block_var_decl::vector_block_var_decl(const std::string& name,
                                                 const vector_block_type& type)
      : var_decl(name, vector_type()), type_(type.bounds(), type.N()) { }

    vector_block_var_decl::vector_block_var_decl(const std::string& name,
                                                 const vector_block_type& type,
                                                 const expression& def)
      : var_decl(name, vector_type(), def),
        type_(type.bounds(), type.N()) { }
  }
}
#endif
