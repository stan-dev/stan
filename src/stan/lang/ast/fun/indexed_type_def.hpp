#ifndef STAN_LANG_AST_FUN_INDEXED_TYPE_DEF_HPP
#define STAN_LANG_AST_FUN_INDEXED_TYPE_DEF_HPP

#include <stan/lang/ast.hpp>
#include <vector>

#include <iostream>

namespace stan {
  namespace lang {

    /*
    *  indexed_type
    *
    *  check each member of vector idxs
    *  multi-idx doesn't change indexed type
    *  uni-idx on array type reduces number of array dimensions
    *  uni-idx on vector/row vector reduces to double
    *  2 uni-idxs on matrix reduces to double
    *  1 uni-idx + 1 multi-idx on matrix reduces to
    *  vector or row_vector depending on position
    *
    */
    bare_expr_type indexed_type(const expression& e,
                                const std::vector<idx>& idxs) {
      // check idxs size, although parser should disallow this
      if (idxs.size() == 0) return e.bare_type();

      // cannot index primitive type
      if (e.bare_type().num_dims() == 0) return ill_formed_type();

      int e_dims = e.bare_type().num_dims();
      int uni_cts = 0;
      for (size_t i = 0; i < idxs.size(); ++i) {
        if (!is_multi_index(idxs[i])) {
          uni_cts++;
        }
      }
      if (uni_cts > e_dims)
        return ill_formed_type();
      
      std::vector<int> slots(e_dims, 0);
      for (size_t i = 0; i < idxs.size(); ++i) {
        if (!is_multi_index(idxs[i])) {
          slots[i] = 1;
        }
      }
      int dims = e_dims - uni_cts;
      if (e.bare_type().base().is_primitive()) {
        if (dims == 0)
          return e.bare_type().base();
        else
          return bare_array_type(e.bare_type().base(), dims);
      }
      if (e.bare_type().base().num_dims() == 1) {  // vector/row_vector
        if (slots[e_dims - 1] == 1) {
          if (dims == 0)
            return double_type();
          else
            return bare_array_type(double_type(), dims);
        } else {  
          --dims;  // vector/row_vector contributes 1 dim to total
          if (dims == 0)
            return e.bare_type().base();
          else
            return bare_array_type(e.bare_type().base(), dims);
        }
      } else {  // matrix type, see table in reference manual for indexing logic
        if (slots[e_dims - 2] == 1 && slots[e_dims - 1] == 1) {
          if (dims == 0)
            return double_type();
          else
            return bare_expr_type(bare_array_type(double_type(), dims));
        } else if (slots[e_dims - 2] == 1) {
          --dims;
          if (dims == 0)
            return row_vector_type();
          else
            return bare_expr_type(bare_array_type(row_vector_type(), dims));
        } else if (slots[e_dims - 1] == 1) {
          --dims;
          if (dims == 0)
            return vector_type();
          else
            return bare_expr_type(bare_array_type(vector_type(), dims));
        } else {
          --dims;
          --dims; // matrix contributes 2 dims to total
          if (dims == 0)
            return matrix_type();
          else
            return bare_expr_type(bare_array_type(matrix_type(), dims));
        }
      }
      return ill_formed_type();
    }
 
  }
}
#endif
