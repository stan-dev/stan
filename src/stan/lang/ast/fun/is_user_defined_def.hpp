#ifndef STAN_LANG_AST_FUN_IS_USER_DEFINED_DEF_HPP
#define STAN_LANG_AST_FUN_IS_USER_DEFINED_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <utility>
#include <vector>

namespace stan {
  namespace lang {

    bool is_user_defined(const fun& fx) {
      return is_user_defined(fx.name_, fx.args_);
    }

    bool is_user_defined(const std::string& name,
                         const std::vector<expression>& args) {
      std::vector<expr_type> arg_types;
      for (size_t i = 0; i <  args.size(); ++i)
        arg_types.push_back(args[i].expression_type());
      function_signature_t sig;
      int matches
        = function_signatures::instance()
        .get_signature_matches(name, arg_types, sig);
      if (matches != 1)
        return false;  // reall shouldn't come up;  throw instead?
      std::pair<std::string, function_signature_t>
        name_sig(name, sig);
      return function_signatures::instance().is_user_defined(name_sig);
    }

  }
}
#endif
