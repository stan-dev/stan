#ifndef STAN_LANG_AST_LOCAL_ARRAY_TYPE_DEF_HPP
#define STAN_LANG_AST_LOCAL_ARRAY_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
namespace lang {

local_array_type::local_array_type()
    : element_type_(ill_formed_type()), array_len_(nil()) {}

local_array_type::local_array_type(const local_var_type& el_type,
                                   const expression& len)
    : element_type_(el_type), array_len_(len) {}

int local_array_type::dims() const {
  int total = 1;
  local_var_type cur_type(element_type_);
  while (cur_type.is_array_type()) {
    total += 1;
    cur_type = cur_type.array_element_type();
  }
  return total;
}

local_array_type::local_array_type(const local_var_type& el_type,
                                   const std::vector<expression>& lens) {
  if (lens.size() == 1) {
    element_type_ = el_type;
    array_len_ = lens[0];
  } else {
    local_array_type tmp(el_type, lens[lens.size() - 1]);
    for (size_t i = lens.size() - 2; i > 0; --i) {
      tmp = local_array_type(tmp, lens[i]);
    }
    element_type_ = tmp;
    array_len_ = lens[0];
  }
}

local_var_type local_array_type::contains() const {
  local_var_type cur_type(element_type_);
  while (cur_type.is_array_type()) {
    cur_type = cur_type.array_element_type();
  }
  return cur_type;
}

local_var_type local_array_type::element_type() const { return element_type_; }

expression local_array_type::array_len() const { return array_len_; }

std::vector<expression> local_array_type::array_lens() const {
  std::vector<expression> result;
  result.push_back(array_len_);
  local_var_type cur_type(element_type_);
  while (cur_type.is_array_type()) {
    result.push_back(cur_type.array_len());
    cur_type = cur_type.array_element_type();
  }
  return result;
}
}  // namespace lang
}  // namespace stan
#endif
