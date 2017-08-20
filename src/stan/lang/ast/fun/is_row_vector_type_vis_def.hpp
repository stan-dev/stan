#ifndef STAN_LANG_AST_FUN_IS_ROW_VECTOR_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_IS_ROW_VECTOR_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    bool is_row_vector_type_vis::operator()(const ill_formed_type& base_type)
      const {
      return false;
    }

    bool is_row_vector_type_vis::operator()(const void_type& base_type)
      const {
      return false;
    }

    bool is_row_vector_type_vis::operator()(const int_type& base_type)
      const {
      return false;
    }

    bool is_row_vector_type_vis::operator()(const double_type& base_type)
      const {
      return false;
    }

    bool is_row_vector_type_vis::operator()(const vector_type& base_type)
      const {
      return false;
    }

    bool is_row_vector_type_vis::operator()(const row_vector_type& base_type)
      const {
      return true;
    }

    bool is_row_vector_type_vis::operator()(const matrix_type& base_type)
      const {
      return false;
    }

  }
}
#endif
