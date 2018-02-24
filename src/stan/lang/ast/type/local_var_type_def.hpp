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

    expression local_var_type::arg1() const {
      var_type_arg1_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    expression local_var_type::arg2() const {
      var_type_arg2_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    local_var_type local_var_type::array_contains() const {
      local_array_base_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    int local_var_type::array_dims() const {
      local_array_dims_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    local_var_type local_var_type::array_element_type() const {
      local_array_element_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    expression local_var_type::array_len() const {
      var_type_array_len_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    std::vector<expression> local_var_type::array_lens() const {
      var_type_array_lens_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    bare_expr_type local_var_type::bare_type() const {
      bare_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    std::string local_var_type::cpp_typename() const {
      cpp_typename_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    bool local_var_type::is_array_type() const {
      return boost::apply_visitor(is_array_type_vis(), var_type_);
    }

    std::string local_var_type::name() const {
      var_type_name_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    int local_var_type::num_dims() const {
      total_dims_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    std::ostream& operator<<(std::ostream& o, const local_var_type& var_type) {
      write_bare_expr_type(o, var_type.bare_type());
      return o;
    }
  }
}
#endif
