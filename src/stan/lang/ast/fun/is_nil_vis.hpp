#ifndef STAN_LANG_AST_FUN_IS_NIL_VIS_HPP
#define STAN_LANG_AST_FUN_IS_NIL_VIS_HPP

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
     * Callback functor for determining if one of the variant types
     * making up an expression is nil.
     */
    struct is_nil_vis : public boost::static_visitor<bool> {
      bool operator()(const nil& x) const;  // NOLINT(runtime/explicit)
      bool operator()(const int_literal& x) const;  // NOLINT(runtime/explicit)
      bool operator()(const double_literal& x) const;  // NOLINT
      bool operator()(const array_expr& x) const;  // NOLINT
      bool operator()(const variable& x) const;  // NOLINT(runtime/explicit)
      bool operator()(const integrate_ode& x) const;  // NOLINT
      bool operator()(const integrate_ode_control& x) const;  // NOLINT
      bool operator()(const fun& x) const;  // NOLINT(runtime/explicit)
      bool operator()(const index_op& x) const;  // NOLINT(runtime/explicit)
      bool operator()(const index_op_sliced& x) const;  // NOLINT
      bool operator()(const conditional_op& x) const;  // NOLINT
      bool operator()(const binary_op& x) const;  // NOLINT(runtime/explicit)
      bool operator()(const unary_op& x) const;  // NOLINT(runtime/explicit)
    };

  }
}
#endif
