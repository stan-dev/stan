#ifndef STAN_LANG_AST_FUN_CPP_TYPENAME_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_CPP_TYPENAME_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>

namespace stan {
  namespace lang {
    cpp_typename_vis::cpp_typename_vis() { }

    std::string cpp_typename_vis::operator()(const block_array_type& x) const {
      // TODO:mitzi - this is wrong - but generator never calls it
      return x.contains().cpp_typename();
    }

    std::string cpp_typename_vis::operator()(const local_array_type& x) const {
      // TODO:mitzi - this is wrong - but generator never calls it
      return x.contains().cpp_typename();
    }

    std::string cpp_typename_vis::operator()(const cholesky_factor_corr_block_type& x) const {
      return "matrix_d";
    }

    std::string cpp_typename_vis::operator()(const cholesky_factor_cov_block_type& x) const {
      return "matrix_d";
    }

    std::string cpp_typename_vis::operator()(const corr_matrix_block_type& x) const {
      return "matrix_d";
    }

    std::string cpp_typename_vis::operator()(const cov_matrix_block_type& x) const {
      return "matrix_d";
    }

    std::string cpp_typename_vis::operator()(const double_block_type& x) const {
      return "real";
    }

    std::string cpp_typename_vis::operator()(const double_type& x) const {
      return "real";
    }

    std::string cpp_typename_vis::operator()(const ill_formed_type& x) const {
      // generator should never call this.
      return "ill_formed";
    }

    std::string cpp_typename_vis::operator()(const int_block_type& x) const {
      return "int";
    }

    std::string cpp_typename_vis::operator()(const int_type& x) const {
      return "int";
    }

    std::string cpp_typename_vis::operator()(const matrix_block_type& x) const {
      return "matrix_d";
    }

    std::string cpp_typename_vis::operator()(const matrix_local_type& x) const {
      return "matrix_d";
    }

    std::string cpp_typename_vis::operator()(const ordered_block_type& x) const {
      return "vector_d";
    }

    std::string cpp_typename_vis::operator()(const positive_ordered_block_type& x) const {
      return "vector_d";
    }

    std::string cpp_typename_vis::operator()(const row_vector_block_type& x) const {
      return "row_vector_d";
    }

    std::string cpp_typename_vis::operator()(const row_vector_local_type& x) const {
      return "row_vector_d";
    }

    std::string cpp_typename_vis::operator()(const simplex_block_type& x) const {
      return "vector_d";
    }

    std::string cpp_typename_vis::operator()(const unit_vector_block_type& x) const {
      return "vector_d";
    }

    std::string cpp_typename_vis::operator()(const vector_block_type& x) const {
      return "vector_d";
    }

    std::string cpp_typename_vis::operator()(const vector_local_type& x) const {
      return "vector_d";
    }
  }
}
#endif
