#ifndef STAN_LANG_AST_FUN_VAR_DECL_SIZE_VIS_SIZE_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_SIZE_VIS_SIZE_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    var_decl_size_vis::var_decl_size_vis() { }

    std::vector<expression>
    var_decl_size_vis::operator()(const nil& /* x */) const {
      return std::vector<expression>();
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const array_block_var_decl& x) const {
      return x.type_.size();
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const array_local_var_decl& x) const {
      // recuse down through array_block_type.element_type_
      return std::vector<expression>();
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const int_block_var_decl& x) const {
      // recuse down through array_block_type.element_type_
      return std::vector<expression>();
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const int_local_var_decl& x) const {
      // recuse down through array_block_type.element_type_
      return std::vector<expression>();
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const double_block_var_decl& x) const {
      return std::vector<expression>();
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const double_local_var_decl& x) const {
      return std::vector<expression>();
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const vector_block_var_decl& x) const {
      return std::vector<expression>(x.type_.N_);
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const vector_local_var_decl& x) const {
      return std::vector<expression>(x.type_.N_);
    }
    
    std::vector<expression>
    var_decl_size_vis::operator()(const row_vector_block_var_decl& x) const {
      return std::vector<expression>(x.type_.N_);
    }
    
    std::vector<expression>
    var_decl_size_vis::operator()(const row_vector_local_var_decl& x) const {
      return std::vector<expression>(x.type_.N_);
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const matrix_block_var_decl& x) const {
      std::vector<expresion> sizes();
      sizes.push_back(x.type_.N_);
      sizes.push_back(x.type_.M_);
      return sizes;
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const matrix_local_var_decl& x) const {
      std::vector<expresion> sizes();
      sizes.push_back(x.type_.N_);
      sizes.push_back(x.type_.M_);
      return sizes;
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const cholesky_factor_block_var_decl& x) const {
      std::vector<expresion> sizes();
      sizes.push_back(x.type_.N_);
      sizes.push_back(x.type_.M_);
      return sizes;
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const cholesky_corr_block_var_decl& x) const {
      return std::vector<expression>(x.type_.K_);
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const cov_matrix_block_var_decl& x) const {
      return std::vector<expression>(x.type_.K_);
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const corr_matrix_block_var_decl& x) const {
      return std::vector<expression>(x.type_.K_);
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const ordered_block_var_decl& x) const {
      return std::vector<expression>(x.type_.K_);
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const positive_ordered_block_var_decl& x) const {
      return std::vector<expression>(x.type_.K_);
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const simplex_block_var_decl& x) const {
      return std::vector<expression>(x.type_.K_);
    }

    std::vector<expression>
    var_decl_size_vis::operator()(const unit_vector_block_var_decl& x) const {
      return std::vector<expression>(x.type_.K_);
    }
  }
}
#endif
