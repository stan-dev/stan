#ifndef STAN_LANG_AST_FUNCTION_ARG_TYPE_HPP
#define STAN_LANG_AST_FUNCTION_ARG_TYPE_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <cstddef>

namespace stan {
  namespace lang {

    /**
     * Structure for a function argument consisting of the expr_type
     * and a boolean used to flag data only arguments.
     */
    struct function_arg_type {
      /**
       * The function argument expr_type.
       */
      expr_type expr_type_;

      /**
       * Boolean true if function argument is data only.
       */
      bool data_only_;

      /**
       * Construct an empty function_argession type.
       */
      function_arg_type();

      /**
       * Construct an function argument type with the specified expr type.
       *
       * @param expr_type function argument expression type
       */
      function_arg_type(const expr_type expr_type);  // NOLINT(runtime/explicit)

      /**
       * Construct an function argument type whith the specified expr type
       * which is a data-only expression.
       *
       * @param expr_type function argument expression type
       * @param num_dims number of dimensions
       */
      function_arg_type(const expr_type expr_type, bool data_only);

      /**
       * Return true if the expr_type of the specified function_arg_type
       * is equal to the expression type of this function_arg_type.
       * Ignore status of bool data_only_.
       *
       * @param fat other function argument type.
       * @return result of equality test.
       */
      bool operator==(const function_arg_type& fat) const;

      /**
       * Return true if the expr_type of the specified function_arg_type
       * is not equal to the expression type of this function_arg_type.
       * Ignore status of bool data_only_.
       *
       * @param fat other function argument type.
       * @return result of equality test.
       */
      bool operator!=(const function_arg_type& fat) const;

      /**
       * Return true if the expr_type of the specified function_arg_type
       * is less than to the expression type of this function_arg_type.
       * Ignore status of bool data_only_.
       * 
       * <p>Types are ordered lexicographically by the value of
       * their function_arg_type(expr_types).
       *
       * @param fat other function argument type.
       * @return result of inequality test.
       */
      bool operator<(const function_arg_type& fat) const;

      /**
       * Return true if the expr_type of the specified function_arg_type
       * is less than or equal to the expression type of this function_arg_type.
       * Ignore status of bool data_only_.
       *
       * @param fat other function argument type.
       * @return result of inequality test.
       */
      bool operator<=(const function_arg_type& fat) const;

      /**
       * Return true if the expr_type of the specified function_arg_type
       * is greater than to the expression type of this function_arg_type.
       * Ignore status of bool data_only_.
       *
       * @param fat other function argument type.
       * @return result of inequality test.
       */
      bool operator>(const function_arg_type& fat) const;

      /**
       * Return true if the expr_type of the specified function_arg_type
       * is greater than or equal to the expression type of this
       * function_arg_type. Ignore status of bool data_only_.
       *
       * @param fat other function argument type.
       * @return result of inequality test.
       */
      bool operator>=(const function_arg_type& fat) const;
    };

  }
}
#endif
