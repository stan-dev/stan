#ifndef STAN_LANG_AST_NODE_EXPRESSION_DEF_HPP
#define STAN_LANG_AST_NODE_EXPRESSION_DEF_HPP


#include <stan/lang/ast/nil.hpp>
#include <stan/lang/ast/node/int_literal.hpp>
#include <stan/lang/ast/node/double_literal.hpp>
#include <stan/lang/ast/node/array_expr.hpp>
#include <stan/lang/ast/node/matrix_expr.hpp>
#include <stan/lang/ast/node/row_vector_expr.hpp>
#include <stan/lang/ast/node/variable.hpp>
#include <stan/lang/ast/node/fun.hpp>
#include <stan/lang/ast/node/integrate_ode.hpp>
#include <stan/lang/ast/node/integrate_ode_control.hpp>
#include <stan/lang/ast/node/algebra_solver.hpp>
#include <stan/lang/ast/node/algebra_solver_control.hpp>
#include <stan/lang/ast/node/map_rect.hpp>
#include <stan/lang/ast/node/index_op.hpp>
#include <stan/lang/ast/node/index_op_sliced.hpp>
#include <stan/lang/ast/node/conditional_op.hpp>
#include <stan/lang/ast/node/binary_op.hpp>
#include <stan/lang/ast/node/unary_op.hpp>

#include <boost/variant/apply_visitor.hpp>
#include <string>

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

    expression::expression(const matrix_expr& expr) : expr_(expr) { }

    expression::expression(const row_vector_expr& expr) : expr_(expr) { }

    expression::expression(const variable& expr) : expr_(expr) { }

    expression::expression(const integrate_ode& expr) : expr_(expr) { }

    expression::expression(const integrate_ode_control& expr) : expr_(expr) { }

    expression::expression(const algebra_solver& expr) : expr_(expr) { }

    expression::expression(const algebra_solver_control& expr) : expr_(expr) { }

    expression::expression(const map_rect& expr) : expr_(expr) { }

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

    bare_expr_type expression::bare_type() const {
      expression_bare_type_vis vis;
      return boost::apply_visitor(vis, expr_);
    }

    int expression::total_dims() const {
      return bare_type().num_dims();
    }

    std::string expression::to_string() const {
      write_expression_vis vis;
      return boost::apply_visitor(vis, expr_);
    }
  }
}
#endif
