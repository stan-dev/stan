#ifndef STAN_LANG_AST_FUN_IS_ARRAY_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_IS_ARRAY_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {
    is_array_type_vis::is_array_type_vis() { }

    bool is_array_type_vis::operator()(const block_array_type& x) const {
      return true;
    }

    bool is_array_type_vis::operator()(const local_array_type& x) const {
      return true;
    }

    bool is_array_type_vis::operator()(const bare_array_type& x) const {
      return true;
    }

    bool is_array_type_vis::operator()(const cholesky_factor_corr_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const cholesky_factor_cov_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const corr_matrix_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const cov_matrix_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const double_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const double_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const ill_formed_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const int_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const int_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const matrix_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const matrix_local_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const matrix_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const ordered_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const positive_ordered_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const row_vector_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const row_vector_local_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const row_vector_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const simplex_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const unit_vector_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const vector_block_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const vector_local_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const vector_type& x) const {
      return false;
    }

    bool is_array_type_vis::operator()(const void_type& x) const {
      return false;
    }
  }
}
#endif
