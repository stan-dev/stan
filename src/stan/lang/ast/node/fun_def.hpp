#ifndef STAN_LANG_AST_NODE_FUN_DEF_HPP
#define STAN_LANG_AST_NODE_FUN_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    fun::fun() { }

    fun::fun(std::string const& name, std::vector<expression> const& args)
      : name_(name), args_(args) { }

  }
}
#endif
