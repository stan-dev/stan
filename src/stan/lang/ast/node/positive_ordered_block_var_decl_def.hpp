#ifndef STAN_LANG_AST_NODE_POSITIVE_ORDERED_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_POSITIVE_ORDERED_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    positive_ordered_block_var_decl::positive_ordered_block_var_decl() { }

    positive_ordered_block_var_decl::positive_ordered_block_var_decl(
                                     const std::string& name,
                                     const positive_ordered_block_type& type)
      : var_decl(name, vector_type()), type_(type.K()) { }

    positive_ordered_block_var_decl::positive_ordered_block_var_decl(
                                     const std::string& name,
                                     const positive_ordered_block_type& type,
                                     const expression& def)
      : var_decl(name, vector_type(), def), type_(type.K()) { }
  }
}
#endif
