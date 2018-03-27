#ifndef STAN_LANG_AST_NODE_EXPRESSION_TYPE_VIS_DEF_HPP
#define STAN_LANG_AST_NODE_EXPRESSION_TYPE_VIS_DEF_HPP

#include <stan/lang/ast.hpp>

namespace stan {
  namespace lang {

    expr_type expression_type_vis::operator()(const nil& /*e*/) const {
      return expr_type();
    }

    expr_type expression_type_vis::operator()(const int_literal& e) const {
      return e.type_;
    }

    expr_type expression_type_vis::operator()(const double_literal& e) const {
      return e.type_;
    }

    expr_type expression_type_vis::operator()(const array_expr& e) const {
      return e.type_;
    }

    expr_type expression_type_vis::operator()(const matrix_expr& e) const {
      return expr_type(matrix_type(), 0);
    }

    expr_type expression_type_vis::operator()(const row_vector_expr& e) const {
      return expr_type(row_vector_type(), 0);
    }

    expr_type expression_type_vis::operator()(const variable& e) const {
      return e.type_;
    }

    expr_type expression_type_vis::operator()(const integrate_ode& e) const {
      return expr_type(double_type(), 2);
    }

    expr_type
    expression_type_vis::operator()(const integrate_ode_control& e) const {
      return expr_type(double_type(), 2);
    }

    expr_type
    expression_type_vis::operator()(const algebra_solver& e) const {
      return expr_type(vector_type(), 0);
    }

    expr_type
    expression_type_vis::operator()(const algebra_solver_control& e) const {
      return expr_type(vector_type(), 0);
    }

    expr_type
    expression_type_vis::operator()(const map_rect& e) const {
      return expr_type(vector_type(), 0);
    }

    expr_type expression_type_vis::operator()(const fun& e) const {
      return e.type_;
    }

    expr_type expression_type_vis::operator()(const index_op& e) const {
      return e.type_;
    }

    expr_type expression_type_vis::operator()(const index_op_sliced& e) const {
      return e.type_;
    }

    expr_type expression_type_vis::operator()(const conditional_op& e) const {
      return e.type_;
    }

    expr_type expression_type_vis::operator()(const binary_op& e) const {
      return e.type_;
    }

    expr_type expression_type_vis::operator()(const unary_op& e) const {
      return e.type_;
    }

  }
}
#endif
