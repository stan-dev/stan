#ifndef STAN_LANG_AST_FUN_LOCAL_ARRAY_ELEMENT_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_LOCAL_ARRAY_ELEMENT_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
namespace lang {
local_array_element_type_vis::local_array_element_type_vis() {}

local_var_type local_array_element_type_vis::operator()(
    const local_array_type& x) const {
  return x.element_type_;
}

local_var_type local_array_element_type_vis::operator()(
    const double_type& x) const {
  return ill_formed_type();
}

local_var_type local_array_element_type_vis::operator()(
    const ill_formed_type& x) const {
  return ill_formed_type();
}

local_var_type local_array_element_type_vis::operator()(
    const int_type& x) const {
  return ill_formed_type();
}

local_var_type local_array_element_type_vis::operator()(
    const matrix_local_type& x) const {
  return ill_formed_type();
}

local_var_type local_array_element_type_vis::operator()(
    const row_vector_local_type& x) const {
  return ill_formed_type();
}

local_var_type local_array_element_type_vis::operator()(
    const vector_local_type& x) const {
  return ill_formed_type();
}
}  // namespace lang
}  // namespace stan
#endif
