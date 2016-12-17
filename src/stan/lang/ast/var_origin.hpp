#ifndef STAN_LANG_AST_VAR_ORIGIN_HPP
#define STAN_LANG_AST_VAR_ORIGIN_HPP

namespace stan {
  namespace lang {

    /**
     * The type of a variable indicating where a variable was
     * declared.   This is a typedef rather than an enum to get around
     * forward declaration issues with enums in header files.
     */
    typedef int var_origin;

    /**
     * Origin of variable is the name of the model.
     */
    const int model_name_origin = 0;

    /**
     * The origin of the variable is the data block.
     */
    const int data_origin = 1;

    /**
     * The origin of the variable is the transformed data block.
     */
    const int transformed_data_origin = 2;

    /**
     * The origin of the variable is the parameter block.
     */
    const int parameter_origin = 3;

    /**
     * The origin of the variable is the transformed parameter block.
     */
    const int transformed_parameter_origin = 4;

    /**
     * The origin of the variable is ???.
     */
    const int derived_origin = 5;

    /**
     * The origin of the variable is as a local variable
     */
    const int local_origin = 6;

    /**
     * The variable arose as a function argument to a non-void
     * function that does not end in _lp or _rng.
     */
    const int function_argument_origin = 7;

    /**
     * The variable arose as an argument to a non-void function with
     * the _lp suffix.
     */
    const int function_argument_origin_lp = 8;

    /**
     * The variable arose as an argument to a non-void function with
     * the _rng suffix.
     */
    const int function_argument_origin_rng = 9;

    /**
     * The variable arose as an argument to a function returning void
     * that does not have the _lp or _rng suffix.
     */
    const int void_function_argument_origin = 10;

    /**
     * The variable arose as an argument to a function returning void
     * with _lp suffix.  function returning void
     */
    const int void_function_argument_origin_lp = 11;

    /**
     * The variable arose as an argument to a function returning void
     * with an _rng suffix.
     */
    const int void_function_argument_origin_rng = 12;

  }
}
#endif
