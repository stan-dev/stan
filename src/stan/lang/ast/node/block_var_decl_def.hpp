#ifndef STAN_LANG_AST_NODE_BLOCK_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_BLOCK_VAR_DECL_DEF_HPP

#include <stan/lang/ast/node/expression_def.hpp>
#include <stan/lang/ast/node/var_decl_def.hpp>
#include <string>

namespace stan {
namespace lang {

block_var_decl::block_var_decl() : type_(ill_formed_type()) {
  this->name_ = "";
  this->bare_type_ = ill_formed_type();
  this->def_ = nil();
}

block_var_decl::block_var_decl(const std::string& name,
                               const block_var_type& type)
    : type_(type) {
  this->name_ = name;
  this->bare_type_ = type.bare_type();
  this->def_ = nil();
}

block_var_decl::block_var_decl(const std::string& name,
                               const block_var_type& type,
                               const expression& def)
    : type_(type) {
  this->name_ = name;
  this->bare_type_ = type.bare_type();
  this->def_ = def;
}

bare_expr_type block_var_decl::bare_type() const { return type_.bare_type(); }

expression block_var_decl::def() const { return this->def_; }

bool block_var_decl::has_def() const { return !is_nil(this->def_); }

std::string block_var_decl::name() const { return this->name_; }

block_var_type block_var_decl::type() const { return type_; }

}  // namespace lang
}  // namespace stan
#endif
