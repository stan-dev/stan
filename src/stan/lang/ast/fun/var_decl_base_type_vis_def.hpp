#ifndef STAN_LANG_AST_FUN_VAR_DECL_BASE_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_BASE_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    var_decl_base_type_vis::var_decl_base_type_vis() { }

    base_var_decl var_decl_base_type_vis::operator()(const nil& /* x */)
      const {
      return base_var_decl();
    }

    base_var_decl var_decl_base_type_vis::operator()(const int_var_decl& x)
      const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(const double_var_decl& x)
      const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(const vector_var_decl& x)
      const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(
                                    const row_vector_var_decl& x) const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(const matrix_var_decl& x)
      const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(
                                    const unit_vector_var_decl& x) const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(
                                     const simplex_var_decl& x) const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(
                                     const ordered_var_decl& x) const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(
                             const positive_ordered_var_decl& x) const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(
                                     const cholesky_factor_var_decl& x) const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(
                                     const cholesky_corr_var_decl& x) const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(
                                     const cov_matrix_var_decl& x) const {
      return x.base_type_;
    }

    base_var_decl var_decl_base_type_vis::operator()(
                                     const corr_matrix_var_decl& x) const {
      return x.base_type_;
    }

  }
}
#endif
