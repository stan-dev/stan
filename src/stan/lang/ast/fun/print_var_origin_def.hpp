#ifndef STAN_LANG_AST_FUN_PRINT_VAR_ORIGIN_DEF_HPP
#define STAN_LANG_AST_FUN_PRINT_VAR_ORIGIN_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    void print_var_origin(std::ostream& o, const var_origin& vo) {
      if (vo == model_name_origin)
        o << "model name";
      else if (vo == data_origin)
        o << "data";
      else if (vo == transformed_data_origin)
        o << "transformed data";
      else if (vo == parameter_origin)
        o << "parameter";
      else if (vo == transformed_parameter_origin)
        o << "transformed parameter";
      else if (vo == derived_origin)
        o << "generated quantities";
      else if (vo == local_origin)
        o << "local";
      else if (vo == function_argument_origin)
        o << "function argument";
      else if (vo == function_argument_origin_lp)
        o << "function argument '_lp' suffixed";
      else if (vo == function_argument_origin_rng)
        o << "function argument '_rng' suffixed";
      else if (vo == void_function_argument_origin)
        o << "void function argument";
      else if (vo == void_function_argument_origin_lp)
        o << "void function argument '_lp' suffixed";
      else if (vo == void_function_argument_origin_rng)
        o << "void function argument '_rng' suffixed";
      else
        o << "UNKNOWN ORIGIN=" << vo;
    }

  }
}
#endif
