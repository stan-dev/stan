#ifndef STAN_LANG_GENERATOR_HAS_ONLY_INT_ARGS_HPP
#define STAN_LANG_GENERATOR_HAS_ONLY_INT_ARGS_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    /**
     * Return true if the specified function has only integer
     * arguments.
     *
     * @param[in] fun function declaration
     * @return bool if the function has only integer arguments
     */
    bool has_only_int_args(const function_decl_def& fun) {
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i)
        if (fun.arg_decls_[i].arg_type_.base_type_ != INT_T)
          return false;
      return true;
    }

  }
}
#endif
