#ifndef STAN_LANG_AST_FUN_INDEXED_TYPE_HPP
#define STAN_LANG_AST_FUN_INDEXED_TYPE_HPP

#include <vector>

namespace stan {
  namespace lang {

    struct expr_type;
    struct expression;
    struct idx;

    /**
     * Return the type of the expression indexed by the generalized
     * index sequence.  Return a type with base type
     * <code>ILL_FORMED_T</code> if there are too many indexes.
     *
     * @param[in] e Expression being indexed.
     * @param[in] idxs Index sequence.
     * @return Type of expression applied to indexes.
     */
    expr_type indexed_type(const expression& e, const std::vector<idx>& idxs);

  }
}
#endif
