#ifndef STAN_LANG_AST_NODE_INDEX_OP_DEF_HPP
#define STAN_LANG_AST_NODE_INDEX_OP_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    index_op::index_op() { }

    index_op::index_op(const expression& expr,
                       const std::vector<std::vector<expression> >& dimss)
      : expr_(expr), dimss_(dimss) {
      infer_type();
    }

    void index_op::infer_type() {
      type_ = infer_type_indexing(expr_, total_dims(dimss_));
    }

  }
}
#endif
