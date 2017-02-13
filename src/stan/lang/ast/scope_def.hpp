#ifndef STAN_LANG_AST_SCOPE_DEF_HPP
#define STAN_LANG_AST_SCOPE_DEF_HPP

#include <stan/lang/ast/origin_block.hpp>
#include <stan/lang/ast/scope.hpp>

namespace stan {
  namespace lang {

    scope::scope()
      : program_block_(model_name_origin), is_local_(false) { }

    scope::scope(const origin_block& program_block)
      : program_block_(program_block), is_local_(false) { }

    scope::scope(const origin_block& program_block,
                           const bool& is_local)
      : program_block_(program_block), is_local_(is_local) { }

    bool scope::is_data_origin() const {
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

    bool scope::is_non_local_parameter_origin() const {
      return  !is_local_
        && program_block_ == parameter_origin;
    }
    bool scope::is_non_local_transformed_parameter_origin() const {
      return  !is_local_
        && program_block_ == transformed_parameter_origin;
    }

    bool scope::is_non_parameter_origin() const {
      return  program_block_ == transformed_parameter_origin
        || program_block_ == local_origin;
    }

    bool scope::is_parameter_origin() const {
      return !is_local_
        && (program_block_ == parameter_origin
            || program_block_ == transformed_parameter_origin);
    }

    bool scope::is_fun_origin() const {
      return program_block_ == function_argument_origin
        || program_block_ == function_argument_origin_lp
        || program_block_ == function_argument_origin_rng
        || program_block_ == void_function_argument_origin
        || program_block_ == void_function_argument_origin_lp
        || program_block_ == void_function_argument_origin_rng;
    }

    bool scope::is_void_function_origin() const {
      return program_block_ == void_function_argument_origin
        || program_block_ == void_function_argument_origin_lp
        || program_block_ == void_function_argument_origin_rng;
    }

    bool scope::is_non_void_function_origin() const {
      return program_block_ == function_argument_origin
        || program_block_ == function_argument_origin_lp
        || program_block_ == function_argument_origin_rng;
    }

    bool scope::allows_assignment() const {
      return !(program_block_ == data_origin
               || program_block_ == parameter_origin);
    }


    bool scope::allows_lp_fun() const {
      return program_block_ == model_name_origin
        || program_block_ == transformed_parameter_origin
        || program_block_ == function_argument_origin_lp
        || program_block_ == void_function_argument_origin_lp;
    }

    bool scope::allows_lp_stmt() const {
      return program_block_ == model_name_origin
        || program_block_ == function_argument_origin_lp
        || program_block_ == void_function_argument_origin_lp;
    }

    bool scope::allows_rng() const {
      return program_block_ == derived_origin
        || program_block_ == transformed_data_origin
        || program_block_ == function_argument_origin_rng
        || program_block_ == void_function_argument_origin_rng;
    }


  }
}
#endif















