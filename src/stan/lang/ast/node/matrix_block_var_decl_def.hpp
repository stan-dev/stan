#ifndef STAN_LANG_AST_NODE_MATRIX_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_MATRIX_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    matrix_block_var_decl::matrix_block_var_decl() { }

    matrix_block_var_decl::matrix_block_var_decl(const std::string& name,
                                                 const matrix_block_type& type)
      : var_decl(name, matrix_type()),
        type_(type.bounds(), type.M(), type.N()) { }

    matrix_block_var_decl::matrix_block_var_decl(const std::string& name,
                                                 const matrix_block_type& type,
                                                 const expression& def)
      : var_decl(name, matrix_type(), def),
        type_(type.bounds(), type.M(), type.N()) { }
  }
}
#endif
