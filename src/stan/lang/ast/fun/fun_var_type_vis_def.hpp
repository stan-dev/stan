#ifndef STAN_LANG_AST_FUN_FUN_VAR_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_FUN_VAR_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    fun_var_type_vis::fun_var_type_vis() { }

    fun_var_type
    fun_var_type_vis::operator()(const nil& /* x */)
      const {
      return fun_var_type();
    }

    fun_var_type
    fun_var_type_vis::operator()(const array_fun_var_decl& x)
      const {
      return fun_var_type(x.bare_type_, x.is_data_);
    }

    fun_var_type
    fun_var_type_vis::operator()(const int_fun_var_decl& x)
      const {
      return fun_var_type(x.bare_type_, x.is_data_);
    }

    fun_var_type
    fun_var_type_vis::operator()(const double_fun_var_decl& x)
      const {
      return fun_var_type(x.bare_type_, x.is_data_);
    }

    fun_var_type
    fun_var_type_vis::operator()(const vector_fun_var_decl& x)
      const {
      return fun_var_type(x.bare_type_, x.is_data_);
    }
    
    fun_var_type
    fun_var_type_vis::operator()(const row_vector_fun_var_decl& x)
      const {
      return fun_var_type(x.bare_type_, x.is_data_);
    }

    fun_var_type
    fun_var_type_vis::operator()(const matrix_fun_var_decl& x)
      const {
      return fun_var_type(x.bare_type_, x.is_data_);
    }
  }
}
#endif
