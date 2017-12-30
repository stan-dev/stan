#ifndef STAN_LANG_AST_FUN_BLOCK_ARRAY_ELEMENT_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_BLOCK_ARRAY_ELEMENT_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {
    block_array_element_type_vis::block_array_element_type_vis() { }

    block_var_type block_array_element_type_vis::operator()(const block_array_type& x) const {
      return x.element_type_;
    }

    block_var_type block_array_element_type_vis::operator()(const cholesky_corr_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const cholesky_factor_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const corr_matrix_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const cov_matrix_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const double_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const ill_formed_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const int_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const matrix_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const ordered_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const positive_ordered_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const row_vector_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const simplex_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const unit_vector_block_type& x) const {
      return ill_formed_type();
    }

    block_var_type block_array_element_type_vis::operator()(const vector_block_type& x) const {
      return ill_formed_type();
    }
  }
}
#endif
