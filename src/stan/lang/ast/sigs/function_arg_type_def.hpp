#ifndef STAN_LANG_AST_FUNCTION_ARG_TYPE_DEF_HPP
#define STAN_LANG_AST_FUNCTION_ARG_TYPE_DEF_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <cstddef>

namespace stan {
  namespace lang {

    function_arg_type::function_arg_type() : data_only_(false) { }

    function_arg_type::function_arg_type(const expr_type expr_type)
      : expr_type_(expr_type), data_only_(false) { }

    function_arg_type::function_arg_type(const expr_type expr_type,
                                         bool data_only)
      : expr_type_(expr_type), data_only_(data_only) { }

    bool function_arg_type::operator==(const function_arg_type& fat) const {
      return expr_type_ == fat.expr_type_;
    }

    bool function_arg_type::operator!=(const function_arg_type& fat) const {
      return expr_type_ != fat.expr_type_;
    }

    bool function_arg_type::operator<(const function_arg_type& fat) const {
      return (expr_type_ < fat.expr_type_);
    }

    bool function_arg_type::operator<=(const function_arg_type& fat) const {
      return (expr_type_ <= fat.expr_type_);
    }

    bool function_arg_type::operator>(const function_arg_type& fat) const {
      return (expr_type_ > fat.expr_type_);
    }

    bool function_arg_type::operator>=(const function_arg_type& fat) const {
      return (expr_type_ >= fat.expr_type_);
    }

  }
}
#endif
