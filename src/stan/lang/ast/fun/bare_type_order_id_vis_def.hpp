#ifndef STAN_LANG_AST_FUN_BARE_TYPE_ORDER_ID_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_BARE_TYPE_ORDER_ID_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

#include <string>

namespace stan {
  namespace lang {
    bare_type_order_id_vis::bare_type_order_id_vis() { }

    std::string bare_type_order_id_vis::operator()(const bare_array_type& x) const {
      return x.oid();
    }

    std::string bare_type_order_id_vis::operator()(const double_type& x) const {
      return x.oid();
    }

    std::string bare_type_order_id_vis::operator()(const ill_formed_type& x) const {
      return x.oid();
    }

    std::string bare_type_order_id_vis::operator()(const int_type& x) const {
      return x.oid();
    }

    std::string bare_type_order_id_vis::operator()(const matrix_type& x) const {
      return x.oid();
    }

    std::string bare_type_order_id_vis::operator()(const row_vector_type& x) const {
      return x.oid();
    }

    std::string bare_type_order_id_vis::operator()(const vector_type& x) const {
      return x.oid();
    }

    std::string bare_type_order_id_vis::operator()(const void_type& x) const {
      return x.oid();
    }
  }
}
#endif
