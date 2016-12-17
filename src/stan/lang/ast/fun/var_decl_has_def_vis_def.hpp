#ifndef STAN_LANG_AST_FUN_VAR_DECL_HAS_DEF_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_HAS_DEF_VIS_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    var_decl_has_def_vis::var_decl_has_def_vis() { }

    bool var_decl_has_def_vis::operator()(const nil& /* x */)
      const {
      return false;  // should not be called
    }

    bool var_decl_has_def_vis::operator()(const int_var_decl& x)
      const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(const double_var_decl& x)
      const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(const vector_var_decl& x)
      const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(
                                    const row_vector_var_decl& x) const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(const matrix_var_decl& x)
      const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(
                                    const unit_vector_var_decl& x) const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(
                                     const simplex_var_decl& x) const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(
                                     const ordered_var_decl& x) const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(
                             const positive_ordered_var_decl& x) const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(
                                     const cholesky_factor_var_decl& x) const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(
                                     const cholesky_corr_var_decl& x) const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(
                                     const cov_matrix_var_decl& x) const {
      return !is_nil(x.def_);
    }

    bool var_decl_has_def_vis::operator()(
                                     const corr_matrix_var_decl& x) const {
      return !is_nil(x.def_);
    }

  }
}
#endif
