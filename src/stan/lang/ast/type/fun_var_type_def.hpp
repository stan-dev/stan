#ifndef STAN_LANG_AST_FUN_VAR_TYPE_DEF_HPP
#define STAN_LANG_AST_FUN_VAR_TYPE_DEF_HPP

#include <stan/lang/ast/type/bare_expr_type.hpp>

namespace stan {
  namespace lang {

    fun_var_type::fun_var_type()
      : bare_type_(ill_formed_type()), is_data_(false) { }

    fun_var_type::fun_var_type(const bare_expr_type& bare_type)
      : bare_type_(bare_type), is_data_(false) { }

    fun_var_type::fun_var_type(const bare_expr_type& bare_type,
                               bool is_data)
      : bare_type_(bare_type), is_data_(is_data) { }


    bool fun_var_type::operator==(const fun_var_type& fvar_type) const {
      return bare_type_.order_id() == fvar_type.bare_type_.order_id();
    }

    bool fun_var_type::operator!=(const fun_var_type& fvar_type) const {
      return bare_type_.order_id() != fvar_type.bare_type_.order_id();
    }

    bool fun_var_type::operator<(const fun_var_type& fvar_type) const {
      return bare_type_.order_id() < fvar_type.bare_type_.order_id();
    }

    bool fun_var_type::operator>(const fun_var_type& fvar_type) const {
      return bare_type_.order_id() > fvar_type.bare_type_.order_id();
    }

    bool fun_var_type::operator<=(const fun_var_type& fvar_type) const {
      return bare_type_.order_id() <= fvar_type.bare_type_.order_id();
    }

    bool fun_var_type::operator>=(const fun_var_type& fvar_type) const {
      return bare_type_.order_id() >= fvar_type.bare_type_.order_id();
    }


    std::ostream& operator<<(std::ostream& o,
                             const fun_var_type& fv_type) {
      if (fv_type.is_data_) o << "data ";
      o << fv_type.bare_type_;
      return o;
    }


  }
}
#endif
