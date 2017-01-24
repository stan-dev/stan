#ifndef STAN_LANG_AST_FUN_PRINT_VAR_ORIGIN_DEF_HPP
#define STAN_LANG_AST_FUN_PRINT_VAR_ORIGIN_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    void print_var_origin(std::ostream& o, const var_origin& vo) {
      if (vo.program_block_ == model_name_origin)
        o << "model name";
      else if (vo.program_block_ == data_origin)
        o << "data";
      else if (vo.program_block_ == transformed_data_origin)
        o << "transformed data";
      else if (vo.program_block_ == parameter_origin)
        o << "parameter";
      else if (vo.program_block_ == transformed_parameter_origin)
        o << "transformed parameter";
      else if (vo.program_block_ == derived_origin)
        o << "generated quantities";
      else if (vo.program_block_ == local_origin)
        o << "local";
      else if (vo.program_block_ == function_argument_origin)
        o << "function argument";
      else if (vo.program_block_ == function_argument_origin_lp)
        o << "function argument '_lp' suffixed";
      else if (vo.program_block_ == function_argument_origin_rng)
        o << "function argument '_rng' suffixed";
      else if (vo.program_block_ == void_function_argument_origin)
        o << "void function argument";
      else if (vo.program_block_ == void_function_argument_origin_lp)
        o << "void function argument '_lp' suffixed";
      else if (vo.program_block_ == void_function_argument_origin_rng)
        o << "void function argument '_rng' suffixed";
      else
        o << "UNKNOWN ORIGIN=" << vo.program_block_ ;
    }

  }
}
#endif
