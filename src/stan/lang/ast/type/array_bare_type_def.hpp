#ifndef STAN_LANG_AST_ARRAY_BARE_TYPE_DEF_HPP
#define STAN_LANG_AST_ARRAY_BARE_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    array_bare_type::array_bare_type()
      : element_type_(ill_formed_type()) { }

    array_bare_type::array_bare_type(const bare_expr_type& el_type)
      : element_type_(el_type) { }
    
    int array_bare_type::array_dims() const {
      int total = 1;
      bare_expr_type cur_type(element_type_);
      while (cur_type.is_array_var_type()) {
        total += 1;
        cur_type = array_bare_type(cur_type).element_type_;
      }
      return total;
    }

    bare_expr_type array_bare_type::contains() const {
      bare_expr_type cur_type(element_type_);
      while (cur_type.is_array_var_type()) {
        cur_type = array_bare_type(cur_type).element_type_;
      }
      return cur_type;
    }
  }
}
#endif
