#ifndef STAN_LANG_AST_BLOCK_VAR_TYPE_DEF_HPP
#define STAN_LANG_AST_BLOCK_VAR_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    block_var_type::block_var_type() : var_type_(ill_formed_type()) { }

    block_var_type::block_var_type(const block_var_type& x)
      : var_type_(x.var_type_) { }

    block_var_type::block_var_type(const block_t& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const ill_formed_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const cholesky_factor_corr_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const cholesky_factor_cov_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const corr_matrix_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const cov_matrix_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const double_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const int_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const matrix_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const ordered_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const positive_ordered_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const row_vector_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const simplex_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const unit_vector_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const vector_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const block_array_type& x)
      : var_type_(x) { }

    expression block_var_type::arg1() const {
      var_type_arg1_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    expression block_var_type::arg2() const {
      var_type_arg2_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    block_var_type block_var_type::array_contains() const {
      block_array_base_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    int block_var_type::array_dims() const {
      block_array_dims_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    block_var_type block_var_type::array_element_type() const {
      block_array_element_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    expression block_var_type::array_len() const {
      var_type_array_len_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    std::vector<expression> block_var_type::array_lens() const {
      var_type_array_lens_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    bare_expr_type block_var_type::bare_type() const {
      bare_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    range block_var_type::bounds() const {
      block_type_bounds_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    std::string block_var_type::cpp_typename() const {
      cpp_typename_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    bool block_var_type::is_array_type() const {
      is_array_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    bool block_var_type::has_def_bounds() const {
      block_type_has_def_bounds_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    std::string block_var_type::name() const {
      var_type_name_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    int block_var_type::num_dims() const {
      total_dims_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    std::ostream& operator<<(std::ostream& o, const block_var_type& var_type) {
      write_block_var_type(o, var_type);
      return o;
    }
  }
}
#endif
