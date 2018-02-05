#ifndef STAN_LANG_AST_FUN_TOTAL_DIMS_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_TOTAL_DIMS_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {
    total_dims_vis::total_dims_vis() { }

    int total_dims_vis::operator()(const block_array_type& x) const {
      return x.dims() + x.contains().num_dims();
    }

    int total_dims_vis::operator()(const local_array_type& x) const {
      return x.dims() + x.contains().num_dims();
    }

    int total_dims_vis::operator()(const bare_array_type& x) const {
      return x.dims() + x.contains().num_dims();
    }

    int total_dims_vis::operator()(const cholesky_factor_corr_block_type& x) const {
      return 2;
    }

    int total_dims_vis::operator()(const cholesky_factor_cov_block_type& x) const {
      return 2;
    }

    int total_dims_vis::operator()(const corr_matrix_block_type& x) const {
      return 2;
    }

    int total_dims_vis::operator()(const cov_matrix_block_type& x) const {
      return 2;
    }

    int total_dims_vis::operator()(const double_block_type& x) const {
      return 0;
    }

    int total_dims_vis::operator()(const double_type& x) const {
      return 0;
    }

    int total_dims_vis::operator()(const ill_formed_type& x) const {
      return 0;
    }

    int total_dims_vis::operator()(const int_block_type& x) const {
      return 0;
    }

    int total_dims_vis::operator()(const int_type& x) const {
      return 0;
    }

    int total_dims_vis::operator()(const matrix_block_type& x) const {
      return 2;
    }

    int total_dims_vis::operator()(const matrix_local_type& x) const {
      return 2;
    }

    int total_dims_vis::operator()(const matrix_type& x) const {
      return 2;
    }

    int total_dims_vis::operator()(const ordered_block_type& x) const {
      return 1;
    }

    int total_dims_vis::operator()(const positive_ordered_block_type& x) const {
      return 1;
    }

    int total_dims_vis::operator()(const row_vector_block_type& x) const {
      return 1;
    }

    int total_dims_vis::operator()(const row_vector_local_type& x) const {
      return 1;
    }

    int total_dims_vis::operator()(const row_vector_type& x) const {
      return 1;
    }

    int total_dims_vis::operator()(const simplex_block_type& x) const {
      return 1;
    }

    int total_dims_vis::operator()(const unit_vector_block_type& x) const {
      return 1;
    }

    int total_dims_vis::operator()(const vector_block_type& x) const {
      return 1;
    }

    int total_dims_vis::operator()(const vector_local_type& x) const {
      return 1;
    }

    int total_dims_vis::operator()(const vector_type& x) const {
      return 1;
    }

    int total_dims_vis::operator()(const void_type& x) const {
      return 0;
    }
  }
}
#endif
