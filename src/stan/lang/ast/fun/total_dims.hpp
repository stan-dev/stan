#ifndef STAN_LANG_AST_FUN_TOTAL_DIMS_HPP
#define STAN_LANG_AST_FUN_TOTAL_DIMS_HPP

#include <cstddef>
#include <vector>

namespace stan {
  namespace lang {

    struct expression;

    /**
     * Return the total number of dimensions when the specified
     * vectors of expressions are concatenated.
     *
     * @param dimss vector of vector of dimension expressions
     */
    std::size_t total_dims(const std::vector<std::vector<expression> >& dimss);

  }
}
#endif
