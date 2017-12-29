#ifndef STAN_LANG_AST_FUN_GET_ARRAY_LOCAL_EL_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_GET_ARRAY_LOCAL_EL_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {

    local_var_type get_array_local_el_type_vis::operator()(const array_local_type& x) const {
      return x.element_type_;
    }

    local_var_type get_array_local_el_type_vis::operator()(const double_type& x) const {
      return ill_formed_type();
    }

    local_var_type get_array_local_el_type_vis::operator()(const ill_formed_type& x) const {
      return ill_formed_type();
    }

    local_var_type get_array_local_el_type_vis::operator()(const int_type& x) const {
      return ill_formed_type();
    }

    local_var_type get_array_local_el_type_vis::operator()(const matrix_local_type& x) const {
      return ill_formed_type();
    }

    local_var_type get_array_local_el_type_vis::operator()(const row_vector_local_type& x) const {
      return ill_formed_type();
    }

    local_var_type get_array_local_el_type_vis::operator()(const vector_local_type& x) const {
      return ill_formed_type();
    }
  }
}
#endif
