#ifndef STAN_LANG_AST_NODE_FUNCTION_DECL_DEF_DEF_HPP
#define STAN_LANG_AST_NODE_FUNCTION_DECL_DEF_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    function_decl_def::function_decl_def() { }

    function_decl_def::function_decl_def(const expr_type& return_type,
                                         const std::string& name,
                                         const std::vector<arg_decl>& arg_decls,
                                         const statement& body)
      : return_type_(return_type), name_(name), arg_decls_(arg_decls),
        body_(body) {
    }

  }
}
#endif
