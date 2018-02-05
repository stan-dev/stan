#ifndef STAN_LANG_AST_FUN_BARE_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_BARE_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {
    bare_type_vis::bare_type_vis() { }

    bare_expr_type bare_type_vis::operator()(const block_array_type& x) const {
      // create innermost element type, then embed in bare_array_type.
      bare_expr_type bare_base = x.contains().bare_type();
      bare_array_type tmp(bare_base);
      bare_expr_type bare_array = bare_expr_type(tmp);
      for (int i=0; i< x.dims() - 1; ++i) {
        tmp = bare_array_type(bare_array);
        bare_array = bare_expr_type(tmp);
      }
      return bare_array;
    }

    bare_expr_type bare_type_vis::operator()(const local_array_type& x) const {
      bare_expr_type bare_base = x.contains().bare_type();
      bare_array_type tmp(bare_base);
      bare_expr_type bare_array = bare_expr_type(tmp);
      for (int i=0; i< x.dims() - 1; ++i) {
        tmp = bare_array_type(bare_array);
        bare_array = bare_expr_type(tmp);
      }
      return bare_array;
    }

    bare_expr_type bare_type_vis::operator()(const cholesky_factor_corr_block_type& x) const {
      return matrix_type();
    }

    bare_expr_type bare_type_vis::operator()(const cholesky_factor_cov_block_type& x) const {
      return matrix_type();
    }

    bare_expr_type bare_type_vis::operator()(const corr_matrix_block_type& x) const {
      return matrix_type();
    }

    bare_expr_type bare_type_vis::operator()(const cov_matrix_block_type& x) const {
      return matrix_type();
    }

    bare_expr_type bare_type_vis::operator()(const double_block_type& x) const {
      return double_type();
    }

    bare_expr_type bare_type_vis::operator()(const double_type& x) const {
      return double_type();
    }

    bare_expr_type bare_type_vis::operator()(const ill_formed_type& x) const {
      return ill_formed_type();
    }

    bare_expr_type bare_type_vis::operator()(const int_block_type& x) const {
      return int_type();
    }

    bare_expr_type bare_type_vis::operator()(const int_type& x) const {
      return int_type();
    }

    bare_expr_type bare_type_vis::operator()(const matrix_block_type& x) const {
      return matrix_type();
    }

    bare_expr_type bare_type_vis::operator()(const matrix_local_type& x) const {
      return matrix_type();
    }

    bare_expr_type bare_type_vis::operator()(const ordered_block_type& x) const {
      return vector_type();
    }

    bare_expr_type bare_type_vis::operator()(const positive_ordered_block_type& x) const {
      return vector_type();
    }

    bare_expr_type bare_type_vis::operator()(const row_vector_block_type& x) const {
      return row_vector_type();
    }

    bare_expr_type bare_type_vis::operator()(const row_vector_local_type& x) const {
      return row_vector_type();
    }

    bare_expr_type bare_type_vis::operator()(const simplex_block_type& x) const {
      return vector_type();
    }

    bare_expr_type bare_type_vis::operator()(const unit_vector_block_type& x) const {
      return vector_type();
    }

    bare_expr_type bare_type_vis::operator()(const vector_block_type& x) const {
      return vector_type();
    }

    bare_expr_type bare_type_vis::operator()(const vector_local_type& x) const {
      return vector_type();
    }
  }
}
#endif
