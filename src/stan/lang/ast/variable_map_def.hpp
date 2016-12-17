#ifndef STAN_LANG_AST_VARIABLE_MAP_DEF_HPP
#define STAN_LANG_AST_VARIABLE_MAP_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>

namespace stan {
  namespace lang {

    bool variable_map::exists(const std::string& name) const {
      return map_.find(name) != map_.end();
    }

    base_var_decl variable_map::get(const std::string& name) const {
      if (!exists(name))
        throw std::invalid_argument("variable does not exist");
      return map_.find(name)->second.first;
    }

    base_expr_type variable_map::get_base_type(const std::string& name) const {
      return get(name).base_type_;
    }

    size_t variable_map::get_num_dims(const std::string& name) const {
      return get(name).dims_.size();
    }

    var_origin variable_map::get_origin(const std::string& name) const {
      if (!exists(name))
        throw std::invalid_argument("variable does not exist");
      return map_.find(name)->second.second;
    }

    void variable_map::add(const std::string& name,
                           const base_var_decl& base_decl,
                           const var_origin& vo) {
      map_[name] = range_t(base_decl, vo);
    }

    void variable_map::remove(const std::string& name) {
      map_.erase(name);
    }

  }
}
#endif
