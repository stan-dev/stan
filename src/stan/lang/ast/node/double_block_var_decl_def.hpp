#ifndef STAN_LANG_AST_NODE_DOUBLE_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_DOUBLE_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    double_block_var_decl::double_block_var_decl() { }

    double_block_var_decl::double_block_var_decl(const std::string& name,
                                                 const double_block_type& type)
      : var_decl(name, double_type()), type_(type.bounds()) { }

    double_block_var_decl::double_block_var_decl(const std::string& name,
                                                 const double_block_type& type,
                                                 const expression& def)
      : var_decl(name, double_type(), def), type_(type.bounds()) { }
  }
}
#endif
