#ifndef STAN_LANG_AST_FUN_BARE_ARRAY_ELEMENT_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_BARE_ARRAY_ELEMENT_TYPE_VIS_DEF_HPP

#include <stan/lang/ast/type/bare_array_type.hpp>
#include <stan/lang/ast/type/double_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_type.hpp>
#include <stan/lang/ast/type/matrix_type.hpp>
#include <stan/lang/ast/type/row_vector_type.hpp>
#include <stan/lang/ast/type/vector_type.hpp>

namespace stan {
namespace lang {
bare_array_element_type_vis::bare_array_element_type_vis() {}

bare_expr_type bare_array_element_type_vis::operator()(
    const bare_array_type& x) const {
  bare_expr_type result = x.element_type_;
  if (x.element_type_.is_data())
    result.set_is_data();
  return result;
}

bare_expr_type bare_array_element_type_vis::operator()(
    const double_type& x) const {
  return ill_formed_type();
}

bare_expr_type bare_array_element_type_vis::operator()(
    const ill_formed_type& x) const {
  return ill_formed_type();
}

bare_expr_type bare_array_element_type_vis::operator()(
    const int_type& x) const {
  return ill_formed_type();
}

bare_expr_type bare_array_element_type_vis::operator()(
    const matrix_type& x) const {
  return ill_formed_type();
}

bare_expr_type bare_array_element_type_vis::operator()(
    const row_vector_type& x) const {
  return ill_formed_type();
}

bare_expr_type bare_array_element_type_vis::operator()(
    const vector_type& x) const {
  return ill_formed_type();
}

bare_expr_type bare_array_element_type_vis::operator()(
    const void_type& x) const {
  return ill_formed_type();
}
}  // namespace lang
}  // namespace stan
#endif
