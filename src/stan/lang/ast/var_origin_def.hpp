#ifndef STAN_LANG_AST_VAR_ORIGIN_DEF_HPP
#define STAN_LANG_AST_VAR_ORIGIN_DEF_HPP

#include <stan/lang/ast/origin_block.hpp>
#include <stan/lang/ast/var_origin.hpp>

namespace stan {
  namespace lang {

    var_origin::var_origin()
      : program_block_(model_name_origin), is_local_(false) { }

    var_origin::var_origin(const origin_block& program_block)
      : program_block_(program_block), is_local_(false) { }

    var_origin::var_origin(const origin_block& program_block,
                           const bool& is_local)
      : program_block_(program_block), is_local_(is_local) { }

    bool var_origin::is_data_origin() const {
      return is_local_
        || (program_block_ == data_origin)
        || (program_block_ == transformed_data_origin)
        || (program_block_ == function_argument_origin)
        || (program_block_ == function_argument_origin_lp)
        || (program_block_ == function_argument_origin_rng)
        || (program_block_ == void_function_argument_origin)
        || (program_block_ == void_function_argument_origin_lp)
        || (program_block_ == void_function_argument_origin_rng);
    }

    bool var_origin::is_parameter_origin() const {
      return !is_local_
        && (program_block_ == parameter_origin
            || program_block_ == transformed_parameter_origin);
    }

    bool var_origin::is_void_function_origin() const {
      return program_block_ == void_function_argument_origin
        || program_block_ == void_function_argument_origin_lp
        || program_block_ == void_function_argument_origin_rng;
    }

    bool var_origin::is_non_void_function_origin() const {
      return program_block_ == function_argument_origin
        || program_block_ == function_argument_origin_lp
        || program_block_ == function_argument_origin_rng;
    }

    bool var_origin::allows_assignment() const {
      return !(program_block_ == data_origin
               || program_block_ == parameter_origin);
    }


    bool var_origin::allows_lp() const {
      return program_block_ == model_name_origin
        || program_block_ == transformed_parameter_origin
        || program_block_ == function_argument_origin_lp
        || program_block_ == void_function_argument_origin_lp;
    }

    bool var_origin::allows_rng() const {
      return program_block_ == derived_origin
        || program_block_ == transformed_data_origin
        || program_block_ == function_argument_origin_rng
        || program_block_ == void_function_argument_origin_rng;
    }


  }
}
#endif















