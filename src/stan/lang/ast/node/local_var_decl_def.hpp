#ifndef STAN_LANG_AST_NODE_LOCAL_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_LOCAL_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    local_var_decl::local_var_decl(const local_var_decl_t& x) : var_decl_(x) { }

    local_var_decl::local_var_decl() : var_decl_(nil()) { }

    local_var_decl::local_var_decl(const nil& x) : var_decl_(x) { }

    local_var_decl::local_var_decl(const int_local_var_decl& x) : var_decl_(x) { }

    local_var_decl::local_var_decl(const double_local_var_decl& x) : var_decl_(x) { }

    local_var_decl::local_var_decl(const vector_local_var_decl& x) : var_decl_(x) { }

    local_var_decl::local_var_decl(const row_vector_local_var_decl& x) : var_decl_(x) { }

    local_var_decl::local_var_decl(const matrix_local_var_decl& x) : var_decl_(x) { }

    bare_expr_type local_var_decl::bare_type() const {
      var_decl_bare_type_vis vis;
      return boost::apply_visitor(vis, var_decl_);
    }

    expression local_var_decl::def() const {
      var_decl_def_vis vis;
      return boost::apply_visitor(vis, var_decl_);
    }

    bool local_var_decl::has_def() const {
      var_decl_has_def_vis vis;
      return boost::apply_visitor(vis, var_decl_);
    }

    std::string local_var_decl::name() const {
      var_decl_name_vis vis;
      return boost::apply_visitor(vis, var_decl_);
    }

    local_var_type local_var_decl::type() const {
      var_decl_local_type_vis vis;
      return boost::apply_visitor(vis, var_decl_);
    }
  }
}
#endif
