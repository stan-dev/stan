#ifndef STAN_LANG_AST_NODE_FUN_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_FUN_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    fun_var_decl::fun_var_decl(const fun_var_decl_t& x) : var_decl_(x) { }

    fun_var_decl::fun_var_decl() : var_decl_(nil()) { }

    fun_var_decl::fun_var_decl(const nil& x) : var_decl_(x) { }

    fun_var_decl::fun_var_decl(const int_fun_var_decl& x) : var_decl_(x) { }

    fun_var_decl::fun_var_decl(const double_fun_var_decl& x) : var_decl_(x) { }

    fun_var_decl::fun_var_decl(const vector_fun_var_decl& x) : var_decl_(x) { }

    fun_var_decl::fun_var_decl(const row_vector_fun_var_decl& x) : var_decl_(x) { }

    fun_var_decl::fun_var_decl(const matrix_fun_var_decl& x) : var_decl_(x) { }

    fun_var_decl::fun_var_decl(const array_fun_var_decl& x) : var_decl_(x) { }

    std::string fun_var_decl::name() const {
      return boost::apply_visitor(var_decl_name_vis(), var_decl_);
    }

    bare_expr_type fun_var_decl::type() const {
      return boost::apply_visitor(var_decl_type_vis(), var_decl_);
    }


  }
}
#endif
