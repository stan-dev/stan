#ifndef STAN_LANG_AST_FUN_INFER_TYPE_INDEXING_HPP
#define STAN_LANG_AST_FUN_INFER_TYPE_INDEXING_HPP

#include <stan/lang/ast/type/bare_expr_type.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <cstddef>

namespace stan {
  namespace lang {

    /**
     * Return the expression type resulting from indexing the
     * specified expression with the specified number of indexes. 
     *
     * @param expr expression being indexed
     * @param num_indexes number of indexes provided
     * @return expression type of indexed expression
     */
    bare_expr_type infer_type_indexing(const expression& e,
                                       std::size_t num_indexes);
  }
}
#endif
