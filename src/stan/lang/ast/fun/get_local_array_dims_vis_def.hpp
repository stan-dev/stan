#ifndef STAN_LANG_AST_FUN_GET_LOCAL_ARRAY_DIMS_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_GET_LOCAL_ARRAY_DIMS_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {
    get_local_array_dims_vis::get_local_array_dims_vis() { }

    int get_local_array_dims_vis::operator()(const local_array_type& x) const {
      return x.dims();
    }

    int get_local_array_dims_vis::operator()(const double_type& x) const {
      return 0;
    }

    int get_local_array_dims_vis::operator()(const ill_formed_type& x) const {
      return 0;
    }

    int get_local_array_dims_vis::operator()(const int_type& x) const {
      return 0;
    }

    int get_local_array_dims_vis::operator()(const matrix_local_type& x) const {
      return 0;
    }

    int get_local_array_dims_vis::operator()(const row_vector_local_type& x) const {
      return 0;
    }

    int get_local_array_dims_vis::operator()(const vector_local_type& x) const {
      return 0;
    }
  }
}
#endif
