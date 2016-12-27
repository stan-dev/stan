#ifndef STAN_LANG_AST_NODE_IDX_DEF_HPP
#define STAN_LANG_AST_NODE_IDX_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    idx::idx() { }

    idx::idx(const uni_idx& i) : idx_(i) { }

    idx::idx(const multi_idx& i) : idx_(i) { }

    idx::idx(const omni_idx& i) : idx_(i) { }

    idx::idx(const lb_idx& i) : idx_(i) { }

    idx::idx(const ub_idx& i) : idx_(i) { }

    idx::idx(const lub_idx& i) : idx_(i) { }

  }
}
#endif
