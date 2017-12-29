#ifndef STAN_LANG_AST_FUN_GET_ARRAY_BARE_EL_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_GET_ARRAY_BARE_EL_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {

    bare_expr_type get_array_bare_el_type_vis::operator()(const array_bare_type& x) const {
      return x.element_type_;
    }

    bare_expr_type get_array_bare_el_type_vis::operator()(const double_type& x) const {
      return ill_formed_type();
    }

    bare_expr_type get_array_bare_el_type_vis::operator()(const ill_formed_type& x) const {
      return ill_formed_type();
    }

    bare_expr_type get_array_bare_el_type_vis::operator()(const int_type& x) const {
      return ill_formed_type();
    }

    bare_expr_type get_array_bare_el_type_vis::operator()(const matrix_type& x) const {
      return ill_formed_type();
    }

    bare_expr_type get_array_bare_el_type_vis::operator()(const row_vector_type& x) const {
      return ill_formed_type();
    }

    bare_expr_type get_array_bare_el_type_vis::operator()(const vector_type& x) const {
      return ill_formed_type();
    }

    bare_expr_type get_array_bare_el_type_vis::operator()(const void_type& x) const {
      return ill_formed_type();
    }
  }
}
#endif
