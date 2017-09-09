#ifndef STAN_LANG_AST_BASE_EXPR_TYPE_DEF_HPP
#define STAN_LANG_AST_BASE_EXPR_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    base_expr_type::base_expr_type() : base_type_(ill_formed_type()) {
    }

    base_expr_type::base_expr_type(const base_expr_type& x)
      : base_type_(x.base_type_), order_id_(x.order_id_) {
    }

    base_expr_type::base_expr_type(const ill_formed_type& x)
      : base_type_(x), order_id_(ill_formed_type::ORDER_ID) {
    }

    base_expr_type::base_expr_type(const void_type& x)
      : base_type_(x), order_id_(void_type::ORDER_ID) {
    }

    base_expr_type::base_expr_type(const int_type& x)
      : base_type_(x), order_id_(int_type::ORDER_ID) {
    }

    base_expr_type::base_expr_type(const double_type& x)
      : base_type_(x), order_id_(double_type::ORDER_ID) {
    }

    base_expr_type::base_expr_type(const vector_type& x)
      : base_type_(x), order_id_(vector_type::ORDER_ID) {
    }

    base_expr_type::base_expr_type(const row_vector_type& x)
      : base_type_(x), order_id_(row_vector_type::ORDER_ID) {
    }

    base_expr_type::base_expr_type(const matrix_type& x)
      : base_type_(x), order_id_(matrix_type::ORDER_ID) {
    }

    bool base_expr_type::is_ill_formed_type() const {
      return order_id_ == ill_formed_type::ORDER_ID;
    }

    bool base_expr_type::is_void_type() const {
      return order_id_ == void_type::ORDER_ID;
    }

    bool base_expr_type::is_int_type() const {
      return order_id_ == int_type::ORDER_ID;
    }

    bool base_expr_type::is_double_type() const {
      return order_id_ == double_type::ORDER_ID;
    }

    bool base_expr_type::is_vector_type() const {
      return order_id_ == vector_type::ORDER_ID;
    }

    bool base_expr_type::is_row_vector_type() const {
      return order_id_ == row_vector_type::ORDER_ID;
    }

    bool base_expr_type::is_matrix_type() const {
      return order_id_ == matrix_type::ORDER_ID;
    }

    bool base_expr_type::operator==(const base_expr_type& base_type) const {
      return order_id_ == base_type.order_id_;
    }

    bool base_expr_type::operator!=(const base_expr_type& base_type) const {
      return order_id_ != base_type.order_id_;
    }

    bool base_expr_type::operator<(const base_expr_type& base_type) const {
      return order_id_ < base_type.order_id_;
    }

    bool base_expr_type::operator>(const base_expr_type& base_type) const {
      return order_id_ > base_type.order_id_;
    }

    bool base_expr_type::operator<=(const base_expr_type& base_type) const {
      return !(order_id_ > base_type.order_id_);
    }

    bool base_expr_type::operator>=(const base_expr_type& base_type) const {
      return !(order_id_ < base_type.order_id_);
    }

  }
}
#endif
