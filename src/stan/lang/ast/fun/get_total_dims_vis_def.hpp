#ifndef STAN_LANG_AST_FUN_GET_TOTAL_DIMS_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_GET_TOTAL_DIMS_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {
    get_total_dims_vis::get_total_dims_vis() { }


    int get_total_dims_vis::operator()(const array_block_type& x) const {
      return array_block_type(x).array_dims() +
             array_block_type(x).contains().num_dims();
    }

    int get_total_dims_vis::operator()(const array_local_type& x) const {
      return array_local_type(x).array_dims() +
             array_local_type(x).contains().num_dims();
    }

    int get_total_dims_vis::operator()(const array_bare_type& x) const {
      return array_bare_type(x).array_dims() +
             array_bare_type(x).contains().num_dims();
    }

    int get_total_dims_vis::operator()(const cholesky_corr_block_type& x) const {
      return 2;
    }

    int get_total_dims_vis::operator()(const cholesky_factor_block_type& x) const {
      return 2;
    }

    int get_total_dims_vis::operator()(const corr_matrix_block_type& x) const {
      return 2;
    }

    int get_total_dims_vis::operator()(const cov_matrix_block_type& x) const {
      return 2;
    }

    int get_total_dims_vis::operator()(const double_block_type& x) const {
      return 0;
    }

    int get_total_dims_vis::operator()(const double_type& x) const {
      return 0;
    }

    int get_total_dims_vis::operator()(const ill_formed_type& x) const {
      return 0;
    }

    int get_total_dims_vis::operator()(const int_block_type& x) const {
      return 0;
    }

    int get_total_dims_vis::operator()(const int_type& x) const {
      return 0;
    }

    int get_total_dims_vis::operator()(const matrix_block_type& x) const {
      return 2;
    }

    int get_total_dims_vis::operator()(const matrix_local_type& x) const {
      return 2;
    }

    int get_total_dims_vis::operator()(const matrix_type& x) const {
      return 2;
    }

    int get_total_dims_vis::operator()(const ordered_block_type& x) const {
      return 1;
    }

    int get_total_dims_vis::operator()(const positive_ordered_block_type& x) const {
      return 1;
    }

    int get_total_dims_vis::operator()(const row_vector_block_type& x) const {
      return 1;
    }

    int get_total_dims_vis::operator()(const row_vector_local_type& x) const {
      return 1;
    }

    int get_total_dims_vis::operator()(const row_vector_type& x) const {
      return 1;
    }

    int get_total_dims_vis::operator()(const simplex_block_type& x) const {
      return 1;
    }

    int get_total_dims_vis::operator()(const unit_vector_block_type& x) const {
      return 1;
    }

    int get_total_dims_vis::operator()(const vector_block_type& x) const {
      return 1;
    }

    int get_total_dims_vis::operator()(const vector_local_type& x) const {
      return 1;
    }

    int get_total_dims_vis::operator()(const vector_type& x) const {
      return 1;
    }

    int get_total_dims_vis::operator()(const void_type& x) const {
      return 0;
    }
  }
}
#endif
