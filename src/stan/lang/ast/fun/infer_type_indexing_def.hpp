#ifndef STAN_LANG_AST_FUN_INFER_TYPE_INDEXING_DEF_HPP
#define STAN_LANG_AST_FUN_INFER_TYPE_INDEXING_DEF_HPP

#include <stan/lang/ast/type/bare_expr_type.hpp>
#include <stan/lang/ast/type/double_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/matrix_type.hpp>
#include <stan/lang/ast/type/row_vector_type.hpp>
#include <stan/lang/ast/type/vector_type.hpp>

namespace stan {
  namespace lang {

    bare_expr_type infer_type_indexing(const bare_expr_type& bare_type,
                                       size_t num_index_dims) {
      
      if (num_index_dims == 0) return bare_type;
      if (num_index_dims > static_cast<size_t>(bare_type.num_dims()))
        return ill_formed_type();

      bare_expr_type tmp = bare_type;
      while (tmp.is_array_type() && num_index_dims > 0) {
        tmp = tmp.array_element_type();
        --num_index_dims;
      }

      if (num_index_dims == 0) return tmp;

      if ((tmp.is_vector_type() || tmp.is_row_vector_type())
          && num_index_dims == 1)
        return double_type();
      
      if (tmp.is_matrix_type() && num_index_dims == 2)
        return double_type();

      if (tmp.is_matrix_type() && num_index_dims == 1)
        return row_vector_type();
      
      return bare_expr_type(ill_formed_type());
    }
  }
}
#endif
