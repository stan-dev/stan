#ifndef STAN_LANG_AST_FUN_VAR_DECL_DIMS_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_DIMS_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    var_decl_dims_vis::var_decl_dims_vis() { }

    std::vector<expression> var_decl_dims_vis::operator()(const nil& /* x */)
      const {
      return std::vector<expression>();  // should not be called
    }

    std::vector<expression> var_decl_dims_vis::operator()(const int_var_decl& x)
      const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                          const double_var_decl& x)
      const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                          const vector_var_decl& x)
      const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                          const row_vector_var_decl& x) const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                          const matrix_var_decl& x)
      const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                          const unit_vector_var_decl& x) const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                          const simplex_var_decl& x) const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                          const ordered_var_decl& x) const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                     const positive_ordered_var_decl& x) const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                     const cholesky_factor_var_decl& x) const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                     const cholesky_corr_var_decl& x) const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                     const cov_matrix_var_decl& x) const {
      return x.dims_;
    }

    std::vector<expression> var_decl_dims_vis::operator()(
                                     const corr_matrix_var_decl& x) const {
      return x.dims_;
    }

  }
}
#endif
