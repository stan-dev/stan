#ifndef STAN_LANG_AST_BARE_EXPR_TYPE_DEF_HPP
#define STAN_LANG_AST_BARE_EXPR_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    bare_expr_type::bare_expr_type()
      : bare_type_(ill_formed_type()) { }

    bare_expr_type::bare_expr_type(const bare_expr_type& x)
      : bare_type_(x.bare_type_), order_id_(x.order_id_) { }

    bare_expr_type::bare_expr_type(const ill_formed_type& x)
      : bare_type_(ill_formed_type()),
        order_id_(ill_formed_type::ORDER_ID) { }

    bare_expr_type::bare_expr_type(const void_type& x)
      : bare_type_(void_type()),
        order_id_(void_type::ORDER_ID) { }

    bare_expr_type::bare_expr_type(const int_type& x)
      : bare_type_(int_type()),
        order_id_(int_type::ORDER_ID) { }

    bare_expr_type::bare_expr_type(const double_type& x)
      : bare_type_(double_type()),
        order_id_(double_type::ORDER_ID) { }

    bare_expr_type::bare_expr_type(const vector_type& x)
      : bare_type_(vector_type()),
        order_id_(vector_type::ORDER_ID) { }

    bare_expr_type::bare_expr_type(const row_vector_type& x)
      : bare_type_(row_vector_type()),
        order_id_(row_vector_type::ORDER_ID) { }

    bare_expr_type::bare_expr_type(const matrix_type& x)
      : bare_type_(matrix_type()),
        order_id_(matrix_type::ORDER_ID) { }

    bare_expr_type::bare_expr_type(const array_bare_type& x)
      : bare_type_(array_bare_type(x.element_type_)),
        order_id_(10 * x.array_dims() + x.contains().order_id_) { }

    bool bare_expr_type::is_ill_formed_type() const {
      return order_id_ == ill_formed_type::ORDER_ID;
    }

    bool bare_expr_type::is_void_type() const {
      return order_id_ == void_type::ORDER_ID;
    }

    bool bare_expr_type::is_int_type() const {
      return order_id_ == int_type::ORDER_ID;
    }

    bool bare_expr_type::is_double_type() const {
      return order_id_ == double_type::ORDER_ID;
    }

    bool bare_expr_type::is_vector_type() const {
      return order_id_ == vector_type::ORDER_ID;
    }

    bool bare_expr_type::is_row_vector_type() const {
      return order_id_ == row_vector_type::ORDER_ID;
    }

    bool bare_expr_type::is_matrix_type() const {
      return order_id_ == matrix_type::ORDER_ID;
    }

    bool bare_expr_type::is_array_var_type() const {
      is_array_var_type_vis vis;
      return boost::apply_visitor(vis, bare_type_);
    }

    bare_expr_type bare_expr_type::get_array_el_type() const {
      get_array_bare_el_type_vis vis;
      return boost::apply_visitor(vis, bare_type_);
    }

    int bare_expr_type::num_dims() const {
      get_total_dims_vis vis;
      return boost::apply_visitor(vis, bare_type_);
    }

    bool bare_expr_type::operator==(const bare_expr_type& bare_type) const {
      return order_id_ == bare_type.order_id_;
    }

    bool bare_expr_type::operator!=(const bare_expr_type& bare_type) const {
      return order_id_ != bare_type.order_id_;
    }

    bool bare_expr_type::operator<(const bare_expr_type& bare_type) const {
      return order_id_ < bare_type.order_id_;
    }

    bool bare_expr_type::operator>(const bare_expr_type& bare_type) const {
      return order_id_ > bare_type.order_id_;
    }

    bool bare_expr_type::operator<=(const bare_expr_type& bare_type) const {
      return order_id_ <= bare_type.order_id_;
    }

    bool bare_expr_type::operator>=(const bare_expr_type& bare_type) const {
      return order_id_ >= bare_type.order_id_;
    }
  }
}
#endif
