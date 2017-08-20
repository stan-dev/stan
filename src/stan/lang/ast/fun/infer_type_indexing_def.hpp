#ifndef STAN_LANG_AST_FUN_INFER_TYPE_INDEXING_DEF_HPP
#define STAN_LANG_AST_FUN_INFER_TYPE_INDEXING_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    expr_type infer_type_indexing(const base_expr_type& expr_base_type,
                                  size_t num_expr_dims,
                                  size_t num_index_dims) {
      if (num_index_dims <= num_expr_dims)
        return expr_type(expr_base_type, num_expr_dims - num_index_dims);
      if (num_index_dims == (num_expr_dims + 1)) {
        if (expr_base_type.is_vector_type() || expr_base_type.is_row_vector_type())
          return expr_type(double_type(), 0U);
        if (expr_base_type.is_matrix_type())
          return expr_type(row_vector_type(), 0U);
      }
      if (num_index_dims == (num_expr_dims + 2))
        if (expr_base_type.is_matrix_type())
          return expr_type(double_type(), 0U);

      // error condition, result expr_type has is_ill_formed() = true
      return expr_type();
    }

    expr_type infer_type_indexing(const expression& expr,
                                  size_t num_index_dims) {
      return infer_type_indexing(expr.expression_type().base_type_,
                                 expr.expression_type().num_dims(),
                                 num_index_dims);
    }




  }
}
#endif
