#ifndef STAN_LANG_AST_FUN_INDEXED_TYPE_DEF_HPP
#define STAN_LANG_AST_FUN_INDEXED_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    expr_type indexed_type(const expression& e, const std::vector<idx>& idxs) {
      expr_type e_type = e.expression_type();

      base_expr_type base_type = e_type.base_type_;
      size_t base_dims = e_type.num_dims_;
      size_t unindexed_dims = base_dims;
      size_t out_dims = 0U;
      size_t i = 0;
      for ( ; unindexed_dims > 0 && i < idxs.size(); ++i, --unindexed_dims)
        if (is_multi_index(idxs[i]))
          ++out_dims;
      if (idxs.size() - i == 0) {
        return expr_type(base_type, out_dims + unindexed_dims);
      } else if (idxs.size() - i == 1) {
        if (base_type == MATRIX_T) {
          if (is_multi_index(idxs[i]))
            return expr_type(MATRIX_T, out_dims);
          else
            return expr_type(ROW_VECTOR_T, out_dims);
        } else if (base_type == VECTOR_T) {
          if (is_multi_index(idxs[i]))
            return expr_type(VECTOR_T, out_dims);
          else
            return expr_type(DOUBLE_T, out_dims);
        } else if (base_type == ROW_VECTOR_T) {
          if (is_multi_index(idxs[i]))
            return expr_type(ROW_VECTOR_T, out_dims);
          else
            return expr_type(DOUBLE_T, out_dims);
        } else {
          return expr_type(ILL_FORMED_T, 0U);
        }
      } else if (idxs.size() - i == 2) {
        if (base_type == MATRIX_T) {
          if (is_multi_index(idxs[i]) && is_multi_index(idxs[i + 1]))
            return expr_type(MATRIX_T, out_dims);
          else if (is_multi_index(idxs[i]))
            return expr_type(VECTOR_T, out_dims);
          else if (is_multi_index(idxs[i + 1]))
            return expr_type(ROW_VECTOR_T, out_dims);
          else
            return expr_type(DOUBLE_T, out_dims);
        } else {
          return expr_type(ILL_FORMED_T, 0U);
        }
      } else {
        return expr_type(ILL_FORMED_T, 0U);
      }
    }

  }
}
#endif
