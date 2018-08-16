#ifndef STAN_LANG_AST_FUN_INFER_TYPE_INDEXING_HPP
#define STAN_LANG_AST_FUN_INFER_TYPE_INDEXING_HPP



#include <stan/lang/ast/base_expr_type.hpp>
#include <stan/lang/ast/expr_type.hpp>
#include <cstddef>

namespace stan {
  namespace lang {

    struct expression;

    /**
     * Return the expression type resulting from indexing an expression
     * of the specified base type and number of dimensions with the
     * specified number of indexes.
     *
     * @param base_type base type of expression being indexed
     * @param dims number of dimensions of the expression being
     * indexed 
     * @param num_indexes number of indexes provided
     * @return expression type of indexed expression
     */
    expr_type infer_type_indexing(const base_expr_type& base_type,
                                  std::size_t dims, std::size_t num_indexes);




    /**
     * Return the expression type resulting from indexing the
     * specified expression with the specified number of indexes. 
     *
     * @param expr expression being indexed
     * @param num_indexes number of indexes provided
     * @return expression type of indexed expression
     */
    expr_type infer_type_indexing(const expression& expr,
                                  std::size_t num_indexes);
  }
}
#endif
