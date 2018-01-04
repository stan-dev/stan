#ifndef STAN_LANG_AST_FUN_INDEXED_TYPE_DEF_HPP
#define STAN_LANG_AST_FUN_INDEXED_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

#include <iostream>

namespace stan {
  namespace lang {

    bare_expr_type indexed_type(const expression& e,
                                const std::vector<idx>& idxs) {

      // std::cout << "e type: " << e.bare_type().order_id() << std::endl;
      if (e.bare_type().is_primitive())
        return ill_formed_type();  // can't index primitive type

      size_t expr_tot_dims = e.bare_type().num_dims();
      if (expr_tot_dims < idxs.size())
        return ill_formed_type(); // too many indexes for this expression

      std::vector<bool> has_idx(expr_tot_dims);
      std::vector<bool> is_multi(expr_tot_dims);
      for (size_t i = 0; i < expr_tot_dims; ++i) {
        if (i < idxs.size()) {
          has_idx[i] = true;
          is_multi[i] = is_multi_index(idxs[i]);
        } else {
          has_idx[i] = false;
          is_multi[i] = false;
        }
      }
      
      bare_expr_type base_type = e.bare_type();
      if (base_type.is_array_type()) base_type = e.bare_type().array_contains();

      bare_expr_type result = base_type;
      // std::cout << "result type: " << result.order_id() << std::endl;

      if (base_type.is_primitive()) {
        // std::cout << "e is array of prim type" << std::endl;
        for (int i=0; i < e.bare_type().array_dims(); ++i) {
          if (is_multi[i] || !(has_idx[i])) result = bare_array_type(result);
          // std::cout << "result type: " << result.order_id() << std::endl;
        }
      }
      return result;


      if (base_type.is_vector_type() || base_type.is_row_vector_type()) {
        // std::cout << "e is (array of) vec type" << std::endl;
        size_t i = expr_tot_dims - 1;
        if (has_idx[i] && !(is_multi[i])) {
          result = double_type();
        }
        for (int i=0; i < e.bare_type().array_dims(); ++i) {
          if (is_multi[i] || !(has_idx[i])) result = bare_array_type(result);
          // std::cout << "result type: " << result.order_id() << std::endl;
        }
        return result;
      }

      if (base_type.is_matrix_type()) {
        // std::cout << "e is (array of) matrix type" << std::endl;
        size_t i_col = expr_tot_dims - 1;
        size_t i_row = expr_tot_dims - 2;
        if (has_idx[i_row] && has_idx[i_col]
            && is_multi[i_row] && is_multi[i_col]) {
          result = matrix_type();
        } else if (has_idx[i_row] && has_idx[i_col] && is_multi[i_row]) {
          result = vector_type();
        } else if (has_idx[i_row] && has_idx[i_col] && is_multi[i_col]) {
          result = row_vector_type();
        } else if (has_idx[i_row] && has_idx[i_col]) {
          result = double_type();
        } else if (has_idx[i_row] && is_multi[i_row]) {
          result = matrix_type();  // matrix w/ single multi-dim idx is matrix!
        } else if (has_idx[i_row]) {
          result = row_vector_type();
        }
        for (int i=0; i < e.bare_type().array_dims(); ++i) {
          if (is_multi[i] || !(has_idx[i])) result = bare_array_type(result);
          // std::cout << "result type: " << result.order_id() << std::endl;
        }
        return result;
      }
      
    }

  }
}
#endif
