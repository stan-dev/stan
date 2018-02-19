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
      bare_expr_type e_bare_type = e.bare_type();

      // check idxs size, although parser should disallow this
      if (idxs.size() == 0) return e_bare_type;  

      size_t tot_dims = e_bare_type.num_dims();
      // cannot index primitive type
      if (tot_dims == 0) return ill_formed_type();
      // too many indexes for expression type
      if (tot_dims < idxs.size()) return ill_formed_type();

      int base_dims = e_bare_type.num_dims() - e_bare_type.array_dims();
      // pad front of idxs so that we can reason about matrices later
      std::vector<idx> idxs_padded;
      multi_idx padding;
      int pad_ct = tot_dims - idxs.size();
      for (int i = 0; i < pad_ct; ++i)
        idxs_padded.push_back(padding);
      for (size_t i = 0; i < idxs.size(); ++i)
        idxs_padded.push_back(idxs[i]);
      
      // int var pos tracks position in vector idxs
      int pos = idxs_padded.size() - 1;
      // check indexes on array dimensions only
      for ( ; pos >= base_dims && e_bare_type.is_array_type(); --pos) {
        if (!is_multi_index(idxs_padded[pos]))
          e_bare_type = e_bare_type.array_element_type();
      }

      // for primitive types, at end of loop, pos == 0
      if (pos < 0) return e_bare_type;

      // vector and matrix types, evaluate indexing on positions 0,1
      // if indexes on array dims are multi-dim, indexed type is an array type
      int ar_dims = e_bare_type.array_dims();

      // compute indexed type for innermost array_element
      if (ar_dims > 0) e_bare_type = e_bare_type.array_contains();

      if (e_bare_type.is_vector_type()
                 || e_bare_type.is_row_vector_type()) {
        if (pos > 0) return ill_formed_type();  // sanity check
        // reduce vector types according to remaining idx
        if (!is_multi_index(idxs_padded[0])) {
          if (ar_dims > 0) return bare_array_type(double_type(), ar_dims);
          else return double_type();
        } else {
          if (ar_dims > 0) return bare_array_type(e_bare_type, ar_dims);
          else return e_bare_type;
        }
      } else if (e_bare_type.is_matrix_type()) {
        if (pos > 1) return ill_formed_type();  // sanity check
        // reduce matrix types according to remaining idx(s)
        if (is_multi_index(idxs_padded[0]) && is_multi_index(idxs_padded[1])) {
          // both row and col idexes unspecified or multi-idx
          if (ar_dims > 0) return bare_array_type(matrix_type(), ar_dims);
          else return matrix_type();
        }
        else if (is_multi_index(idxs_padded[1]) || pad_ct == 1) {
          // no index specified for row, multi-idx on col
          if (ar_dims > 0) return bare_array_type(row_vector_type(), ar_dims);
          else return row_vector_type();
        }
        else if (is_multi_index(idxs_padded[0])  && pad_ct == 0) {
          // specified uni-index on row, multi-idx on col
          if (ar_dims > 0) return bare_array_type(vector_type(), ar_dims);
          else return vector_type();
        }
        else {
          if (ar_dims > 0) return bare_array_type(double_type(), ar_dims);
          else return double_type();
        }
      }
      return ill_formed_type();
    }
 
  }
}
#endif
