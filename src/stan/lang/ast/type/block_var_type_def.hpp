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

    block_var_type::block_var_type(const cholesky_corr_block_type& x)
      : var_type_(x) { }

    block_var_type::block_var_type(const cholesky_factor_block_type& x)
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

    block_var_type::block_var_type(const array_block_type& x)
      : var_type_(x) { }

    bool block_var_type::is_array_var_type() const {
      is_array_var_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    block_var_type block_var_type::get_array_el_type() const {
      get_array_block_el_type_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }

    int block_var_type::num_dims() const {
      get_total_dims_vis vis;
      return boost::apply_visitor(vis, var_type_);
    }
  }
}
#endif
