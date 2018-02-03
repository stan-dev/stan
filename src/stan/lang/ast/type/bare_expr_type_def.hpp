#ifndef STAN_LANG_AST_BARE_EXPR_TYPE_DEF_HPP
#define STAN_LANG_AST_BARE_EXPR_TYPE_DEF_HPP

#include <stan/lang/ast/type/bare_array_type.hpp>
#include <stan/lang/ast/type/bare_expr_type.hpp>
#include <stan/lang/ast/type/double_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_type.hpp>
#include <stan/lang/ast/type/matrix_type.hpp>
#include <stan/lang/ast/type/row_vector_type.hpp>
#include <stan/lang/ast/type/vector_type.hpp>
#include <stan/lang/ast/type/void_type.hpp>

#include <stan/lang/ast/fun/write_bare_expr_type.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    bare_expr_type::bare_expr_type()
      : bare_type_(ill_formed_type()), is_data_(false) { }

    bare_expr_type::bare_expr_type(const bare_expr_type& x)
      : bare_type_(x.bare_type_), is_data_(false) { }

    bare_expr_type::bare_expr_type(const bare_t& x)
      : bare_type_(x), is_data_(false) { }

    bare_expr_type::bare_expr_type(const ill_formed_type& x)
      : bare_type_(ill_formed_type()), is_data_(false) { }

    bare_expr_type::bare_expr_type(const void_type& x)
      : bare_type_(void_type()), is_data_(false) { }

    bare_expr_type::bare_expr_type(const int_type& x)
      : bare_type_(int_type()), is_data_(false) { }

    bare_expr_type::bare_expr_type(const double_type& x)
      : bare_type_(double_type()), is_data_(false) { }

    bare_expr_type::bare_expr_type(const vector_type& x)
      : bare_type_(vector_type()), is_data_(false) { }

    bare_expr_type::bare_expr_type(const row_vector_type& x)
      : bare_type_(row_vector_type()), is_data_(false) { }

    bare_expr_type::bare_expr_type(const matrix_type& x)
      : bare_type_(matrix_type()), is_data_(false) { }

    bare_expr_type::bare_expr_type(const bare_array_type& x)
      : bare_type_(bare_array_type(x.element_type_)), is_data_(false) { }

    bool bare_expr_type::is_ill_formed_type() const {
      return order_id() == ill_formed_type().oid();
    }

    bool bare_expr_type::is_void_type() const {
      return order_id() == void_type().oid();
    }

    bool bare_expr_type::is_int_type() const {
      return order_id() == int_type().oid();
    }

    bool bare_expr_type::is_double_type() const {
      return order_id() == double_type().oid();
    }

    bool bare_expr_type::is_vector_type() const {
      return order_id() == vector_type().oid();
    }

    bool bare_expr_type::is_row_vector_type() const {
      return order_id() == row_vector_type().oid();
    }

    bool bare_expr_type::is_matrix_type() const {
      return order_id() == matrix_type().oid();
    }

    bool bare_expr_type::is_primitive() const {
      return order_id() == int_type().oid()
        || order_id() == double_type().oid();
    }

    bool bare_expr_type::is_array_type() const {
      is_array_type_vis vis;
      return boost::apply_visitor(vis, bare_type_);
    }

    bool bare_expr_type::is_data() const {
      return is_data_;
    }

    bare_expr_type bare_expr_type::array_element_type() const {
      bare_array_element_type_vis vis;
      return boost::apply_visitor(vis, bare_type_);
    }

    bare_expr_type bare_expr_type::array_contains() const {
      bare_array_base_type_vis vis;
      return boost::apply_visitor(vis, bare_type_);
    }

    int bare_expr_type::array_dims() const {
      bare_array_dims_vis vis;
      return boost::apply_visitor(vis, bare_type_);
    }

    int bare_expr_type::num_dims() const {
      total_dims_vis vis;
      return boost::apply_visitor(vis, bare_type_);
    }

    std::string bare_expr_type::order_id() const {
      bare_type_order_id_vis vis;
      return boost::apply_visitor(vis, bare_type_);
    }

    void bare_expr_type::set_is_data(bool f) {
      is_data_ = f;
    }

    bool bare_expr_type::operator==(const bare_expr_type& bare_type) const {
      return order_id() == bare_type.order_id();
    }

    bool bare_expr_type::operator!=(const bare_expr_type& bare_type) const {
      return order_id() != bare_type.order_id();
    }

    bool bare_expr_type::operator<(const bare_expr_type& bare_type) const {
      return order_id() < bare_type.order_id();
    }

    bool bare_expr_type::operator>(const bare_expr_type& bare_type) const {
      return order_id() > bare_type.order_id();
    }

    bool bare_expr_type::operator<=(const bare_expr_type& bare_type) const {
      return order_id() <= bare_type.order_id();
    }

    bool bare_expr_type::operator>=(const bare_expr_type& bare_type) const {
      return order_id() >= bare_type.order_id();
    }

    std::ostream& operator<<(std::ostream& o, const bare_expr_type& bare_type) {
      write_bare_expr_type(o, bare_type);
      return o;
    }
  }
}
#endif
