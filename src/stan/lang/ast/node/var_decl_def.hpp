#ifndef STAN_LANG_AST_NODE_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    var_decl::var_decl(const var_decl_t& decl) : decl_(decl) { }

    var_decl::var_decl() : decl_(nil()) { }

    var_decl::var_decl(const nil& decl) : decl_(decl) { }

    var_decl::var_decl(const int_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const double_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const vector_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const row_vector_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const matrix_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const unit_vector_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const simplex_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const ordered_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const positive_ordered_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const cholesky_factor_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const cholesky_corr_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const cov_matrix_var_decl& decl) : decl_(decl) { }

    var_decl::var_decl(const corr_matrix_var_decl& decl) : decl_(decl) { }

    std::string var_decl::name() const {
      return boost::apply_visitor(name_vis(), decl_);
    }

    base_var_decl var_decl::base_decl() const {
      return boost::apply_visitor(var_decl_base_type_vis(), decl_);
    }

    std::vector<expression> var_decl::dims() const {
      return boost::apply_visitor(var_decl_dims_vis(), decl_);
    }

    bool var_decl::has_def() const {
      return boost::apply_visitor(var_decl_has_def_vis(), decl_);
    }

    expression var_decl::def() const {
      return boost::apply_visitor(var_decl_def_vis(), decl_);
    }

  }
}
#endif
