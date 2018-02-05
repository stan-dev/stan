#ifndef STAN_LANG_AST_FUN_BLOCK_ARRAY_DIMS_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_BLOCK_ARRAY_DIMS_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {
    block_array_dims_vis::block_array_dims_vis() { }

    int block_array_dims_vis::operator()(const block_array_type& x) const {
      return x.dims();
    }

    int block_array_dims_vis::operator()(const cholesky_factor_corr_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const cholesky_factor_cov_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const corr_matrix_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const cov_matrix_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const double_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const ill_formed_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const int_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const matrix_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const ordered_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const positive_ordered_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const row_vector_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const simplex_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const unit_vector_block_type& x) const {
      return 0;
    }

    int block_array_dims_vis::operator()(const vector_block_type& x) const {
      return 0;
    }
  }
}
#endif
