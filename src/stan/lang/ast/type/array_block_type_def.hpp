#ifndef STAN_LANG_AST_ARRAY_BLOCK_TYPE_DEF_HPP
#define STAN_LANG_AST_ARRAY_BLOCK_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    array_block_type::array_block_type() { }

    array_block_type::array_block_type(const block_var_type& el_type,
                                       const expression& len)
      : element_type_(el_type), array_len_(len) { }
    
    size_t array_block_type::array_dims() const {
      size_t total = 1;
      block_var_type cur_type(element_type_);
      while (cur_type.is_array_var_type()) {
        total += 1;
        cur_type = cur_type.get_array_el_type();
      }
      return total;
    }

    block_var_type array_block_type::contains() const {
      block_var_type cur_type(element_type_);
      while (cur_type.is_array_var_type()) {
        cur_type = cur_type.get_array_el_type();
      }
      return cur_type;
    }

    expression array_block_type::array_len() const {
      return array_len_;
    }
  }
}
#endif
