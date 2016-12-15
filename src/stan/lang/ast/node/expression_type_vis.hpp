#ifndef STAN_LANG_AST_NODE_EXPRESSION_TYPE_VIS_HPP
#define STAN_LANG_AST_NODE_EXPRESSION_TYPE_VIS_HPP

#include <stan/lang/ast/expr_type.hpp>
#include <boost/variant/static_visitor.hpp>

namespace stan {
  namespace lang {

    struct nil;
    struct int_literal;
    struct double_literal;
    struct array_expr;
    struct variable;
    struct fun;
    struct integrate_ode;
    struct integrate_ode_control;
    struct index_op;
    struct index_op_sliced;
    struct conditional_op;
    struct binary_op;
    struct unary_op;

    /**
     * Callback functor for calculating expression types from the
     * variant types making up an expression.
     */
    struct expression_type_vis : public boost::static_visitor<expr_type> {
      expr_type operator()(const nil& e) const;
      expr_type operator()(const int_literal& e) const;
      expr_type operator()(const double_literal& e) const;
      expr_type operator()(const array_expr& e) const;
      expr_type operator()(const variable& e) const;
      expr_type operator()(const fun& e) const;
      expr_type operator()(const integrate_ode& e) const;
      expr_type operator()(const integrate_ode_control& e) const;
      expr_type operator()(const index_op& e) const;
      expr_type operator()(const index_op_sliced& e) const;
      expr_type operator()(const conditional_op& e) const;
      expr_type operator()(const binary_op& e) const;
      expr_type operator()(const unary_op& e) const;
    };

  }
}
#endif
