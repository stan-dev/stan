#ifndef STAN_LANG_AST_FUNCTION_ARG_TYPE_DEF_HPP
#define STAN_LANG_AST_FUNCTION_ARG_TYPE_DEF_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/sigs/function_arg_type.hpp>
#include <cstddef>
#include <ostream>

namespace stan {
  namespace lang {

    function_arg_type::function_arg_type() : data_only_(false) { }

    function_arg_type::function_arg_type(const expr_type& e_type)
      : expr_type_(e_type), data_only_(false) { }

    function_arg_type::function_arg_type(const expr_type& e_type,
                                         bool data_only)
      : expr_type_(e_type), data_only_(data_only) { }

    bool function_arg_type::operator==(const function_arg_type& fa_type) const {
      return expr_type_ == fa_type.expr_type_;
    }

    bool function_arg_type::operator!=(const function_arg_type& fa_type) const {
      return expr_type_ != fa_type.expr_type_;
    }

    bool function_arg_type::operator<(const function_arg_type& fa_type) const {
      return (expr_type_ < fa_type.expr_type_);
    }

    bool function_arg_type::operator<=(const function_arg_type& fa_type) const {
      return (expr_type_ <= fa_type.expr_type_);
    }

    bool function_arg_type::operator>(const function_arg_type& fa_type) const {
      return (expr_type_ > fa_type.expr_type_);
    }

    bool function_arg_type::operator>=(const function_arg_type& fa_type) const {
      return (expr_type_ >= fa_type.expr_type_);
    }

    std::ostream& operator<<(std::ostream& o,
                             const function_arg_type& fa_type) {
      if (fa_type.data_only_) o << "data ";
      o << fa_type.expr_type_;
      return o;
    }

  }
}
#endif
