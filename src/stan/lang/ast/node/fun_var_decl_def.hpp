#ifndef STAN_LANG_AST_NODE_FUN_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_FUN_VAR_DECL_DEF_HPP


#include <stan/lang/ast/node/expression_def.hpp>
#include <stan/lang/ast/node/var_decl_def.hpp>
#include <string>

namespace stan {
  namespace lang {

    fun_var_decl::fun_var_decl()
      : type_(ill_formed_type()) {
      this->name_ = "";
      this->bare_type_ = ill_formed_type();
      this->def_ = nil();
    }

    fun_var_decl::fun_var_decl(const std::string& name,
                                   const bare_expr_type& type)
      : type_(type) {
      this->name_ = name;
      this->bare_type_ = type.bare_type_;
      this->def_ = nil();
    }

    fun_var_decl::fun_var_decl(const std::string& name,
                                   const bare_expr_type& type,
                                   const expression& def)
      : type_(type) {
      this->name_ = name;
      this->bare_type_ = type.bare_type_;
      this->def_ = def;
    }
    

    bare_expr_type fun_var_decl::bare_type() const {
      return this->bare_type_;
    }

    expression fun_var_decl::def() const {
      return this->def_;
    }

    bool fun_var_decl::has_def() const {
      return !is_nil(this->def_);
    }

    std::string fun_var_decl::name() const {
      return this->name_;
    }

    bare_expr_type fun_var_decl::type() const {
      return type_;
    }
  }
}
#endif
