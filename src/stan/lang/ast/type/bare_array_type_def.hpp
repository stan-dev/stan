#ifndef STAN_LANG_AST_BARE_ARRAY_TYPE_DEF_HPP
#define STAN_LANG_AST_BARE_ARRAY_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    bare_array_type::bare_array_type()
      : element_type_(ill_formed_type()) { }

    bare_array_type::bare_array_type(const bare_expr_type& el_type)
      : element_type_(el_type) { }
    
    bare_array_type::bare_array_type(const bare_expr_type& el_type,
                                     size_t num_dims) {
      if (num_dims == 0
          || el_type.is_ill_formed_type()
          || el_type.is_array_type()) {
        element_type_ = ill_formed_type();
        return;
      }
      if (num_dims == 1) {
        element_type_ = el_type;
        return;
      }
      bare_array_type bat(el_type);
      bare_expr_type  bet(bat);
      for (size_t i = 1; i < num_dims; ++i) {
        bet = bare_expr_type(bat);
        bat = bare_array_type(bet);
      }
      element_type_ = bet;
    }
    
    int bare_array_type::dims() const {
      if (element_type_.is_ill_formed_type()) return 0;
      int total = 1;
      bare_expr_type cur_type(element_type_);
      while (cur_type.is_array_type()) {
        total += 1;
        cur_type = cur_type.array_element_type();
      }
      return total;
    }

    bare_expr_type bare_array_type::contains() const {
      bare_expr_type cur_type(element_type_);
      while (cur_type.is_array_type()) {
        cur_type = cur_type.array_element_type();
      }
      return cur_type;
    }

    std::string bare_array_type::oid() const {
      std::string oid = std::string("array_") + element_type_.order_id();
      return oid;
    }

  }
}
#endif
