#ifndef STAN_LANG_AST_FUN_BARE_ARRAY_DIMS_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_BARE_ARRAY_DIMS_VIS_DEF_HPP

#include <stan/lang/ast/type/bare_array_type.hpp>
#include <stan/lang/ast/type/double_type.hpp>
#include <stan/lang/ast/type/ill_formed_type.hpp>
#include <stan/lang/ast/type/int_type.hpp>
#include <stan/lang/ast/type/matrix_type.hpp>
#include <stan/lang/ast/type/row_vector_type.hpp>
#include <stan/lang/ast/type/vector_type.hpp>

namespace stan {
  namespace lang {
    bare_array_dims_vis::bare_array_dims_vis() { }

    int bare_array_dims_vis::operator()(const bare_array_type& x) const {
      return x.dims();
    }

    int bare_array_dims_vis::operator()(const double_type& x) const {
      return 0;
    }

    int bare_array_dims_vis::operator()(const ill_formed_type& x) const {
      return 0;
    }

    int bare_array_dims_vis::operator()(const int_type& x) const {
      return 0;
    }

    int bare_array_dims_vis::operator()(const matrix_type& x) const {
      return 0;
    }

    int bare_array_dims_vis::operator()(const row_vector_type& x) const {
      return 0;
    }

    int bare_array_dims_vis::operator()(const vector_type& x) const {
      return 0;
    }

    int bare_array_dims_vis::operator()(const void_type& x) const {
      return 0;
    }
  }
}
#endif
