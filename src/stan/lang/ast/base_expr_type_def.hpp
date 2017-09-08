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

    template <typename T>
    base_expr_type::base_expr_type(const T& base_type)
      : base_type_(base_type), order_id_(T::ORDER_ID) {
    }

    bool base_expr_type::is_ill_formed_type() const {
      is_ill_formed_type_vis v;
      return boost::apply_visitor(v, base_type_);
    }

    bool base_expr_type::is_void_type() const {
      is_void_type_vis v;
      return boost::apply_visitor(v, base_type_);
    }

    bool base_expr_type::is_int_type() const {
      is_int_type_vis v;
      return boost::apply_visitor(v, base_type_);
    }

    bool base_expr_type::is_double_type() const {
      is_double_type_vis v;
      return boost::apply_visitor(v, base_type_);
    }

    bool base_expr_type::is_vector_type() const {
      is_vector_type_vis v;
      return boost::apply_visitor(v, base_type_);
    }

    bool base_expr_type::is_row_vector_type() const {
      is_row_vector_type_vis v;
      return boost::apply_visitor(v, base_type_);
    }

    bool base_expr_type::is_matrix_type() const {
      is_matrix_type_vis v;
      return boost::apply_visitor(v, base_type_);
    }

    bool base_expr_type::operator==(const base_expr_type& base_type) const {
      return ((is_ill_formed_type() &&
           base_type.is_ill_formed_type())
          || (is_void_type() &&
              base_type.is_void_type())
          || (is_int_type() &&
              base_type.is_int_type())
          || (is_double_type() &&
              base_type.is_double_type())
          || (is_vector_type() &&
              base_type.is_vector_type())
          || (is_row_vector_type() &&
              base_type.is_row_vector_type())
          || (is_matrix_type() &&
              base_type.is_matrix_type()));
    }

    bool base_expr_type::operator!=(const base_expr_type& base_type) const {
      return !(*this == base_type);
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
