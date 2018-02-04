#ifndef STAN_LANG_AST_LOCAL_ARRAY_TYPE_DEF_HPP
#define STAN_LANG_AST_LOCAL_ARRAY_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    local_array_type::local_array_type()
      : element_type_(ill_formed_type()), array_len_(nil()) { }

    local_array_type::local_array_type(const local_var_type& el_type,
                                       const expression& len)
    : element_type_(el_type), array_len_(len) { }

    int local_array_type::dims() const {
      int total = 1;
      local_var_type cur_type(element_type_);
      while (cur_type.is_array_type()) {
        total += 1;
        cur_type = cur_type.array_element_type();
      }
      return total;
    }

    local_var_type local_array_type::contains() const {
      local_var_type cur_type(element_type_);
      while (cur_type.is_array_type()) {
        cur_type = cur_type.array_element_type();
      }
      return cur_type;
    }

    local_var_type local_array_type::element_type() const {
      return element_type_;
    }

    expression local_array_type::array_len() const {
      return array_len_;
    }
  }
}
#endif
