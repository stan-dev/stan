#ifndef STAN_LANG_AST_FUNCTION_ARG_TYPE_HPP
#define STAN_LANG_AST_FUNCTION_ARG_TYPE_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <cstddef>

namespace stan {
  namespace lang {

    /**
     * Structure for a function argument consisting of the
     * `expr_type` and a boolean used to flag
     * data only arguments.
     */
    struct function_arg_type {
      /**
       * The function argument `expr_type`.
       */
      expr_type expr_type_;

      /**
       * Boolean true if function argument is data only.
       */
      bool data_only_;

      /**
       * Construct an empty `function_arg_type`.
       */
      function_arg_type();

      /**
       * Construct a `function_arg_type` with the specified
       * `expr_type`.
       *
       * @param e_type function argument expression type
       */
      explicit function_arg_type(const expr_type& e_type);

      /**
       * Construct a `function_arg_type` with the specified
       * `expr_type` which is a data-only expression.
       *
       * @param e_type function argument expression type
       * @param data_only true if argument has prefix qualifier `data`
       */
      function_arg_type(const expr_type& e_type, bool data_only);

      /**
       * Return true if the `expr_type` of the specified
       * `function_arg_type` is equal to the `expr_type`
       * of this `function_arg_type`.
       * Ignore status of bool `data_only`_.
       *
       * @param fa_type other function argument type.
       * @return result of equality test.
       */
      bool operator==(const function_arg_type& fa_type) const;

      /**
       * Return true if the `expr_type` of the specified
       * `function_arg_type` is not equal to the
       * `expr_type` of this `function_arg_type</
       * Ignore status of bool `data_only_`.
       *
       * @param fa_type other function argument type.
       * @return result of not equals test.
       */
      bool operator!=(const function_arg_type& fa_type) const;

      /**
       * Return true if the `expr_type` of the specified
       * `function_arg_type` is less than to
       * the `expr_type` of this `function_arg_type`.
       * Ignore status of bool `data_only_`.
       * 
       * <p>Types are ordered lexicographically by the value of
       * their function_arg_type(expr_types).
       *
       * @param fa_type other function argument type.
       * @return result of less than test.
       */
      bool operator<(const function_arg_type& fa_type) const;

      /**
       * Return true if the `expr_type` of the specified
       * `function_arg_type` is less than or equal to the
       * `expr_type` of this `function_arg_type`.
       * Ignore status of bool `data_only_`.
       *
       * @param fa_type other function argument type.
       * @return result of less than or equals test.
       */
      bool operator<=(const function_arg_type& fa_type) const;

      /**
       * Return true if the `expr_type` of the specified
       * `function_arg_type` is greater than the
       * `expr_type` of this `function_arg_type`.
       * Ignore status of bool `data_only_`.
       *
       * @param fa_type other function argument type.
       * @return result of greater than test.
       */
      bool operator>(const function_arg_type& fa_type) const;

      /**
       * Return true if the `expr_type` of the specified
       * `function_arg_type` is greater than or equal to the
       * `expr_type` of this `function_arg_type`.
       * Ignore status of bool `data_only_`.
       *
       * @param fa_type other function argument type.
       * @return result of greater than or equals test.
       */
      bool operator>=(const function_arg_type& fa_type) const;
    };

  }
}
#endif
