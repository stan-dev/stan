#ifndef STAN_LANG_AST_FUN_INDEXED_TYPE_DEF_HPP
#define STAN_LANG_AST_FUN_INDEXED_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

namespace stan {
  namespace lang {

    bare_expr_type indexed_type(const expression& e, const std::vector<idx>& idxs) {
      bare_expr_type e_type = e.expression_type();

      size_t base_dims = e_type.num_dims();
      size_t unindexed_dims = base_dims;
      size_t out_dims = 0U;
      size_t i = 0;
      // counts down total dims, taking into account slices 
      for ( ; unindexed_dims > 0 && i < idxs.size(); ++i, --unindexed_dims)
        if (is_multi_index(idxs[i]))
          ++out_dims;
      // indexed expression has total dims: out_dims + unindexed_dims
      // case:
      // e.bare.is_array_type() && total dims == array_dims return contained type
      // e.bare.is_array_type() && total dims < array_dims return contained type
      // e.bare.is_array_type() && total dims > array_dims - depends on contained type
      // e.bare.is_vector_type() && total dims == 1, return double_type
      // e.bare.is_row_vector_type() && total dims == 1, return double_type
      // e.bare.is_matrix_type() && total dims == 1 row_vector_type
      // e.bare.is_matrix_type() && total dims == 2 return double_type


      // if (idxs.size() - i == 0) {
      //   return expr_type(base_type, out_dims + unindexed_dims);
      // } else if (idxs.size() - i == 1) {
      //   if (base_type.is_matrix_type()) {
      //     if (is_multi_index(idxs[i]))
      //       return expr_type(matrix_type(), out_dims);
      //     else
      //       return expr_type(row_vector_type(), out_dims);
      //   } else if (base_type.is_vector_type()) {
      //     if (is_multi_index(idxs[i]))
      //       return expr_type(vector_type(), out_dims);
      //     else
      //       return expr_type(double_type(), out_dims);
      //   } else if (base_type.is_row_vector_type()) {
      //     if (is_multi_index(idxs[i]))
      //       return expr_type(row_vector_type(), out_dims);
      //     else
      //       return expr_type(double_type(), out_dims);
      //   } else {
      //     return expr_type(ill_formed_type(), 0U);
      //   }
      // } else if (idxs.size() - i == 2) {
      //   if (base_type.is_matrix_type()) {
      //     if (is_multi_index(idxs[i]) && is_multi_index(idxs[i + 1]))
      //       return expr_type(matrix_type(), out_dims);
      //     else if (is_multi_index(idxs[i]))
      //       return expr_type(vector_type(), out_dims);
      //     else if (is_multi_index(idxs[i + 1]))
      //       return expr_type(row_vector_type(), out_dims);
      //     else
      //       return expr_type(double_type(), out_dims);
      //   } else {
      //     return expr_type(ill_formed_type(), 0U);
      //   }
      // } else {
      //   return expr_type(ill_formed_type(), 0U);
      // }

      return bare_expr_type(ill_formed_type());
    }

  }
}
#endif
