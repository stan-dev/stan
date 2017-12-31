#ifndef STAN_LANG_AST_FUN_VAR_DECL_LOCAL_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_VAR_DECL_LOCAL_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    var_decl_local_type_vis::var_decl_local_type_vis() { }

    local_var_type
    var_decl_local_type_vis::operator()(const nil& /* x */)
      const {
      return ill_formed_type();
    }

    local_var_type
    var_decl_local_type_vis::operator()(const array_local_var_decl& x)
      const {
      return x.type_;
    }

    local_var_type
    var_decl_local_type_vis::operator()(const int_local_var_decl& x)
      const {
      return int_type();
    }

    local_var_type
    var_decl_local_type_vis::operator()(const double_local_var_decl& x)
      const {
      return double_type();
    }

    local_var_type
    var_decl_local_type_vis::operator()(const vector_local_var_decl& x)
      const {
      return x.type_;
    }
    
    local_var_type
    var_decl_local_type_vis::operator()(const row_vector_local_var_decl& x)
      const {
      return x.type_;
    }

    local_var_type
    var_decl_local_type_vis::operator()(const matrix_local_var_decl& x)
      const {
      return x.type_;
    }
  }
}
#endif
