#ifndef STAN_LANG_AST_FUN_VAR_DECL_IS_DATA_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_IS_DATA_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    var_decl_is_data_vis::var_decl_is_data_vis() { }

    bool
    var_decl_is_data_vis::operator()(const nil& /* x */)
      const {
      return false;
    }

    bool
    var_decl_is_data_vis::operator()(const array_fun_var_decl& x)
      const {
      return x.is_data_;
    }

    bool
    var_decl_is_data_vis::operator()(const int_fun_var_decl& x)
      const {
      return x.is_data_;
    }

    bool
    var_decl_is_data_vis::operator()(const double_fun_var_decl& x)
      const {
      return x.is_data_;
    }

    bool
    var_decl_is_data_vis::operator()(const vector_fun_var_decl& x)
      const {
      return x.is_data_;
    }
    
    bool
    var_decl_is_data_vis::operator()(const row_vector_fun_var_decl& x)
      const {
      return x.is_data_;
    }

    bool
    var_decl_is_data_vis::operator()(const matrix_fun_var_decl& x)
      const {
      return x.is_data_;
    }
  }
}
#endif
