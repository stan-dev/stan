#ifndef STAN_LANG_AST_NODE_EXPRESSION_HPP
#define STAN_LANG_AST_NODE_EXPRESSION_HPP

#include <stan/lang/ast/expr_type.hpp>

#include <boost/variant/recursive_variant.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    struct nil;
    struct int_literal;
    struct double_literal;
    struct array_expr;
    struct matrix_expr;
    struct row_vector_expr;
    struct variable;
    struct fun;
    struct integrate_ode;
    struct integrate_ode_control;
    struct index_op;
    struct index_op_sliced;
    struct conditional_op;
    struct binary_op;
    struct unary_op;

    struct expression {
      typedef boost::variant<boost::recursive_wrapper<nil>,
                             boost::recursive_wrapper<int_literal>,
                             boost::recursive_wrapper<double_literal>,
                             boost::recursive_wrapper<array_expr>,
                             boost::recursive_wrapper<matrix_expr>,
                             boost::recursive_wrapper<row_vector_expr>,
                             boost::recursive_wrapper<variable>,
                             boost::recursive_wrapper<integrate_ode>,
                             boost::recursive_wrapper<integrate_ode_control>,
                             boost::recursive_wrapper<fun>,
                             boost::recursive_wrapper<index_op>,
                             boost::recursive_wrapper<index_op_sliced>,
                             boost::recursive_wrapper<conditional_op>,
                             boost::recursive_wrapper<binary_op>,
                             boost::recursive_wrapper<unary_op> >
      expression_t;

      expression();
      expression(const expression& e);

      // template <typename Expr> expression(const Expr& expr);
      expression(const nil& expr);  // NOLINT(runtime/explicit)
      expression(const int_literal& expr);  // NOLINT(runtime/explicit)
      expression(const double_literal& expr);  // NOLINT(runtime/explicit)
      expression(const array_expr& expr);  // NOLINT(runtime/explicit)
      expression(const matrix_expr& expr);  // NOLINT(runtime/explicit)
      expression(const row_vector_expr& expr);  // NOLINT(runtime/explicit)
      expression(const variable& expr);  // NOLINT(runtime/explicit)
      expression(const fun& expr);  // NOLINT(runtime/explicit)
      expression(const integrate_ode& expr);  // NOLINT(runtime/explicit)
      expression(const integrate_ode_control& expr);  // NOLINT
      expression(const index_op& expr);  // NOLINT(runtime/explicit)
      expression(const index_op_sliced& expr);  // NOLINT(runtime/explicit)
      expression(const conditional_op& expr);  // NOLINT(runtime/explicit)
      expression(const binary_op& expr);  // NOLINT(runtime/explicit)
      expression(const unary_op& expr);  // NOLINT(runtime/explicit)
      expression(const expression_t& expr_);  // NOLINT(runtime/explicit)

      expr_type expression_type() const;
      int total_dims() const;

      expression& operator+=(const expression& rhs);
      expression& operator-=(const expression& rhs);
      expression& operator*=(const expression& rhs);
      expression& operator/=(const expression& rhs);

      expression_t expr_;
    };

  }
}
#endif
