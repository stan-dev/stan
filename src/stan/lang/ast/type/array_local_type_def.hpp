#ifndef STAN_LANG_AST_ARRAY_LOCAL_TYPE_DEF_HPP
#define STAN_LANG_AST_ARRAY_LOCAL_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    array_local_type::array_local_type()
      : element_type_(ill_formed_type()) { }

    array_local_type::array_local_type(const local_var_type& el_type,
                                       const expression& len)
    : element_type_(el_type), array_len_(len) { }

    size_t array_local_type::array_dims() const {
      size_t total = 1;
      local_var_type cur_type(element_type_);
      while (cur_type.is_array_var_type()) {
        total += 1;
        cur_type = cur_type.get_array_el_type();
      }
      return total;
    }

    local_var_type array_local_type::contains() const {
      local_var_type cur_type(element_type_);
      while (cur_type.is_array_var_type()) {
        cur_type = cur_type.get_array_el_type();
      }
      return cur_type;
    }

    expression array_local_type::array_len() const {
      return array_len_;
    }
  }
}
#endif
