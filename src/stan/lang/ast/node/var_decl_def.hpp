#ifndef STAN_LANG_AST_NODE_VAR_DECL_DEF_HPP
#define STAN_LANG_AST_NODE_VAR_DECL_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    var_decl::var_decl() { }

    var_decl::var_decl(const std::string& name)
      : name_(name), bare_type_(ill_formed_type()), is_data_(false) {  }

    var_decl::var_decl(const std::string& name,
                       const bare_expr_type& type)
      : name_(name), bare_type_(type), is_data_(false) { }

    var_decl::var_decl(const std::string& name,
                       const bare_expr_type& type,
                       const expression& def)
      : name_(name), bare_type_(type), is_data_(false), def_(def) {  }

    bare_expr_type var_decl::bare_type() const {
      return bare_type_;
    }

    expression var_decl::def() const {
      return def_;
    }

    bool var_decl::is_data() const {
      return is_data_;
    }

    std::string var_decl::name() const {
      return name_;
    }
    
    void var_decl::set_is_data(bool flag) {
      is_data_ = flag;
    }
  }
}

#endif
