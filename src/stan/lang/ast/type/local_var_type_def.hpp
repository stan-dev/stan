#ifndef STAN_LANG_AST_LOCAL_VAR_TYPE_DEF_HPP
#define STAN_LANG_AST_LOCAL_VAR_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    local_var_type::local_var_type() : var_type_(ill_formed_type()) { }

    local_var_type::local_var_type(const local_var_type& x)
      : var_type_(x.var_type_) { }

    local_var_type::local_var_type(const local_t& x)
      : var_type_(x) { }

    local_var_type::local_var_type(const ill_formed_type& x)
      : var_type_(x) { }

    local_var_type::local_var_type(const int_type& x)
      : var_type_(x) { }

    local_var_type::local_var_type(const double_type& x)
      : var_type_(x) { }

    local_var_type::local_var_type(const vector_local_type& x)
      : var_type_(x) { }

    local_var_type::local_var_type(const row_vector_local_type& x)
      : var_type_(x) { }

    local_var_type::local_var_type(const matrix_local_type& x)
      : var_type_(x) { }

    local_var_type::local_var_type(const local_array_type& x)
      : var_type_(x) { }

    bool local_var_type::is_array_type() const {
      return boost::apply_visitor(is_array_type_vis(), var_type_);
    }

    local_var_type local_var_type::array_element_type() const {
      get_local_array_element_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    local_var_type local_var_type::array_contains() const {
      get_local_array_base_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    int local_var_type::array_dims() const {
      get_local_array_dims_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    int local_var_type::num_dims() const {
      get_total_dims_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }
  }
}
#endif
