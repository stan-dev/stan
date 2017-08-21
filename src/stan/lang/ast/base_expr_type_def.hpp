#ifndef STAN_LANG_AST_BASE_EXPR_TYPE_DEF_HPP
#define STAN_LANG_AST_BASE_EXPR_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    base_expr_type::base_expr_type()
      : base_type_(ill_formed_type())  {
    }

    base_expr_type::base_expr_type(const base_expr_type_t&
                                   base_type)
      : base_type_(base_type) {
    }

    base_expr_type::base_expr_type(const ill_formed_type&
                                   base_type)
      : base_type_(base_type) {
    }

    base_expr_type::base_expr_type(const void_type&
                                   base_type)
      : base_type_(base_type) {
    }

    base_expr_type::base_expr_type(const int_type&
                                   base_type)
      : base_type_(base_type) {
    }

    base_expr_type::base_expr_type(const double_type&
                                   base_type)
      : base_type_(base_type) {
    }

    base_expr_type::base_expr_type(const vector_type&
                                   base_type)
      : base_type_(base_type) {
    }

    base_expr_type::base_expr_type(const row_vector_type&
                                   base_type)
      : base_type_(base_type) {
    }

    base_expr_type::base_expr_type(const matrix_type&
                                   base_type)
      : base_type_(base_type) {
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
      if ((is_ill_formed_type() &&
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
              base_type.is_matrix_type()))
        return true;
      return false;
    }

    bool base_expr_type::operator!=(const base_expr_type& base_type) const {
      if ((is_ill_formed_type() &&
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
              base_type.is_matrix_type()))
        return false;
      return true;
    }

  }
}
#endif
