#ifndef STAN_LANG_AST_FUN_INFER_TYPE_INDEXING_DEF_HPP
#define STAN_LANG_AST_FUN_INFER_TYPE_INDEXING_DEF_HPP

#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/type/bare_expr_type.hpp>
#include <stan/lang/ast/type/double_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/matrix_type.hpp>
#include <stan/lang/ast/type/row_vector_type.hpp>
#include <stan/lang/ast/type/vector_type.hpp>

namespace stan {
  namespace lang {

    bare_expr_type infer_type_indexing(const expression& e,
                                       size_t num_index_dims) {
      bare_expr_type bare_type = e.bare_type();
      size_t num_expr_dims = bare_type.num_dims();

      while (bare_type.is_array_type()
             && num_index_dims > 1) {
        bare_type = bare_type.array_element_type();
        num_expr_dims--;
        num_index_dims--;
      }

      if (num_expr_dims < num_index_dims) 
        return bare_expr_type(ill_formed_type());

      if (bare_type.is_array_type())
        return bare_type.array_element_type();
      
      if ((bare_type.is_row_vector_type()
          || bare_type.is_vector_type())
          && num_index_dims == 1)
        return bare_expr_type(double_type());

      if (bare_type.is_matrix_type()
          &&  num_index_dims == 1)
        return bare_expr_type(row_vector_type());

      if (bare_type.is_matrix_type()
          &&  num_index_dims == 1)
        return bare_expr_type(double_type());
      
      return bare_expr_type(ill_formed_type());
    }
  }
}
#endif
