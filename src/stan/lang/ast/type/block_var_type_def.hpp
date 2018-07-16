#ifndef STAN_LANG_AST_BLOCK_VAR_TYPE_DEF_HPP
#define STAN_LANG_AST_BLOCK_VAR_TYPE_DEF_HPP

#include <stan/lang/ast/type/block_array_type.hpp>
#include <stan/lang/ast/type/cholesky_factor_corr_block_type.hpp>
#include <stan/lang/ast/type/cholesky_factor_cov_block_type.hpp>
#include <stan/lang/ast/type/corr_matrix_block_type.hpp>
#include <stan/lang/ast/type/cov_matrix_block_type.hpp>
#include <stan/lang/ast/type/double_block_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_block_type.hpp>
#include <stan/lang/ast/type/matrix_block_type.hpp>
#include <stan/lang/ast/type/ordered_block_type.hpp>
#include <stan/lang/ast/type/positive_ordered_block_type.hpp>
#include <stan/lang/ast/type/row_vector_block_type.hpp>
#include <stan/lang/ast/type/simplex_block_type.hpp>
#include <stan/lang/ast/type/unit_vector_block_type.hpp>
#include <stan/lang/ast/type/vector_block_type.hpp>

#include <stan/lang/ast/fun/bare_type_vis.hpp>
#include <stan/lang/ast/fun/block_type_bounds_vis.hpp>
#include <stan/lang/ast/fun/block_type_is_specialized_vis.hpp>
#include <stan/lang/ast/fun/block_type_params_total_vis.hpp>
#include <stan/lang/ast/fun/var_type_arg1_vis.hpp>
#include <stan/lang/ast/fun/var_type_arg2_vis.hpp>
#include <stan/lang/ast/fun/var_type_name_vis.hpp>
#include <stan/lang/ast/fun/write_block_var_type.hpp>

#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/get.hpp>

#include <ostream>
#include <string>
#include <vector>

namespace stan {
namespace lang {

block_var_type::block_var_type() : var_type_(ill_formed_type()) {}

block_var_type::block_var_type(const block_var_type& x)
    : var_type_(x.var_type_) {}

block_var_type::block_var_type(const block_t& x) : var_type_(x) {}

block_var_type::block_var_type(const ill_formed_type& x) : var_type_(x) {}

block_var_type::block_var_type(const cholesky_factor_corr_block_type& x)
    : var_type_(x) {}

block_var_type::block_var_type(const cholesky_factor_cov_block_type& x)
    : var_type_(x) {}

block_var_type::block_var_type(const corr_matrix_block_type& x)
    : var_type_(x) {}

block_var_type::block_var_type(const cov_matrix_block_type& x) : var_type_(x) {}

block_var_type::block_var_type(const double_block_type& x) : var_type_(x) {}

block_var_type::block_var_type(const int_block_type& x) : var_type_(x) {}

block_var_type::block_var_type(const matrix_block_type& x) : var_type_(x) {}

block_var_type::block_var_type(const ordered_block_type& x) : var_type_(x) {}

block_var_type::block_var_type(const positive_ordered_block_type& x)
    : var_type_(x) {}

block_var_type::block_var_type(const row_vector_block_type& x) : var_type_(x) {}

block_var_type::block_var_type(const simplex_block_type& x) : var_type_(x) {}

block_var_type::block_var_type(const unit_vector_block_type& x)
    : var_type_(x) {}

block_var_type::block_var_type(const vector_block_type& x) : var_type_(x) {}

block_var_type::block_var_type(const block_array_type& x) : var_type_(x) {}

expression block_var_type::arg1() const {
  var_type_arg1_vis vis;
  return boost::apply_visitor(vis, var_type_);
}

expression block_var_type::arg2() const {
  var_type_arg2_vis vis;
  return boost::apply_visitor(vis, var_type_);
}

block_var_type block_var_type::array_contains() const {
  if (boost::get<stan::lang::block_array_type>(&var_type_)) {
    block_array_type vt = boost::get<stan::lang::block_array_type>(var_type_);
    return vt.contains();
  }
  return ill_formed_type();
}

int block_var_type::array_dims() const {
  if (boost::get<stan::lang::block_array_type>(&var_type_)) {
    block_array_type vt = boost::get<stan::lang::block_array_type>(var_type_);
    return vt.dims();
  }
  return 0;
}

block_var_type block_var_type::array_element_type() const {
  if (boost::get<stan::lang::block_array_type>(&var_type_)) {
    block_array_type vt = boost::get<stan::lang::block_array_type>(var_type_);
    return vt.element_type();
  }
  return ill_formed_type();
}

expression block_var_type::array_len() const {
  if (boost::get<stan::lang::block_array_type>(&var_type_)) {
    block_array_type vt = boost::get<stan::lang::block_array_type>(var_type_);
    return vt.array_len();
  }
  return expression(nil());
}

std::vector<expression> block_var_type::array_lens() const {
  if (boost::get<stan::lang::block_array_type>(&var_type_)) {
    block_array_type vt = boost::get<stan::lang::block_array_type>(var_type_);
    return vt.array_lens();
  }
  return std::vector<expression>();
}

bare_expr_type block_var_type::bare_type() const {
  bare_type_vis vis;
  return boost::apply_visitor(vis, var_type_);
}

range block_var_type::bounds() const {
  block_type_bounds_vis vis;
  return boost::apply_visitor(vis, var_type_);
}

bool block_var_type::has_def_bounds() const {
  if (this->bounds().has_low() || this->bounds().has_high())
    return true;
  return false;
}
  
block_var_type block_var_type::innermost_type() const {
  if (boost::get<stan::lang::block_array_type>(&var_type_)) {
    block_array_type vt = boost::get<stan::lang::block_array_type>(var_type_);
    return vt.contains();
  }
  return var_type_;
}
  
bool block_var_type::is_array_type() const {
  if (boost::get<stan::lang::block_array_type>(&var_type_))
    return true;
  return false;
}

bool block_var_type::is_constrained() const {
  return has_def_bounds() || is_specialized();
}

bool block_var_type::is_specialized() const {
  block_type_is_specialized_vis vis;
  return boost::apply_visitor(vis, var_type_);
}

std::string block_var_type::name() const {
  var_type_name_vis vis;
  return boost::apply_visitor(vis, var_type_);
}

int block_var_type::num_dims() const {
  return this->bare_type().num_dims();
}

expression block_var_type::params_total() const {
  block_type_params_total_vis vis;
  return boost::apply_visitor(vis, var_type_);
}

std::ostream& operator<<(std::ostream& o, const block_var_type& var_type) {
  write_block_var_type(o, var_type);
  return o;
}
}  // namespace lang
}  // namespace stan
#endif
