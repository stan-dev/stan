#ifndef STAN_LANG_AST_NODE_EXPRESSION_DEF_HPP
#define STAN_LANG_AST_NODE_EXPRESSION_DEF_HPP

#include <stan/lang/ast.hpp>


namespace stan {
  namespace lang {

    expression::expression()
      : expr_(nil()) {
    }

    expression::expression(const expression& e)
      : expr_(e.expr_) {
    }

    expression::expression(const expression_t& expr) : expr_(expr) { }

    expression::expression(const nil& expr) : expr_(expr) { }

    expression::expression(const int_literal& expr) : expr_(expr) { }

    expression::expression(const double_literal& expr) : expr_(expr) { }

    expression::expression(const array_expr& expr) : expr_(expr) { }

    expression::expression(const variable& expr) : expr_(expr) { }

    expression::expression(const integrate_ode& expr) : expr_(expr) { }

    expression::expression(const integrate_ode_control& expr) : expr_(expr) { }

    expression::expression(const fun& expr) : expr_(expr) { }

    expression::expression(const index_op& expr) : expr_(expr) { }

    expression::expression(const index_op_sliced& expr) : expr_(expr) { }

    expression::expression(const conditional_op& expr) : expr_(expr) { }

    expression::expression(const binary_op& expr) : expr_(expr) { }

    expression::expression(const unary_op& expr) : expr_(expr) { }

    expression& expression::operator+=(const expression& rhs) {
      expr_ = binary_op(expr_, "+", rhs);
      return *this;
    }

    expression& expression::operator-=(const expression& rhs) {
      expr_ = binary_op(expr_, "-", rhs);
      return *this;
    }

    expression& expression::operator*=(const expression& rhs) {
      expr_ = binary_op(expr_, "*", rhs);
      return *this;
    }

    expression& expression::operator/=(const expression& rhs) {
      expr_ = binary_op(expr_, "/", rhs);
      return *this;
    }

    expr_type expression::expression_type() const {
      expression_type_vis vis;
      return boost::apply_visitor(vis, expr_);
    }

    int expression::total_dims() const {
      int sum = expression_type().num_dims_;
      if (expression_type().type() == VECTOR_T)
        ++sum;
      if (expression_type().type() == ROW_VECTOR_T)
        ++sum;
      if (expression_type().type() == MATRIX_T)
        sum += 2;
      return sum;
    }

  }
}
#endif
