#ifndef STAN_LANG_AST_NODE_VARIABLE_DIMS_DEF_HPP
#define STAN_LANG_AST_NODE_VARIABLE_DIMS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    variable_dims::variable_dims() { }

    variable_dims::variable_dims(std::string const& name,
                                 std::vector<expression> const& dims)
      : name_(name), dims_(dims) { }

  }
}
#endif
