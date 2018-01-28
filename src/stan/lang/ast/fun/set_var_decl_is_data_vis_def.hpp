#ifndef STAN_LANG_AST_FUN_SET_VAR_DECL_IS_DATA_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_SET_VAR_DECL_IS_DATA_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    set_var_decl_is_data_vis::set_var_decl_is_data_vis() { }

    bool set_var_decl_is_data_vis::operator()
      (const nil& /* x */) const {
      return false;
    }

    bool set_var_decl_is_data_vis::operator()
      (array_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (array_local_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (array_fun_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (int_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (int_local_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (int_fun_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (double_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (double_local_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (double_fun_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (vector_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (vector_local_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (vector_fun_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }
    
    bool set_var_decl_is_data_vis::operator()
      (row_vector_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }
    
    bool set_var_decl_is_data_vis::operator()
      (row_vector_local_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }
    
    bool set_var_decl_is_data_vis::operator()
      (row_vector_fun_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (matrix_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (matrix_local_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (matrix_fun_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (cholesky_factor_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (cholesky_corr_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (cov_matrix_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (corr_matrix_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (ordered_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (positive_ordered_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (simplex_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }

    bool set_var_decl_is_data_vis::operator()
      (unit_vector_block_var_decl& x) const {
      x.set_is_data(true);
      return true;
    }
  }
}
#endif
