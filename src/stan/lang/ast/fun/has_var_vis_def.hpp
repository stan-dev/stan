#ifndef STAN_LANG_AST_FUN_HAS_VAR_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_HAS_VAR_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <iostream>

namespace stan {
  namespace lang {

    has_var_vis::has_var_vis(const variable_map& var_map)
      : var_map_(var_map) {
    }

    bool has_var_vis::operator()(const nil& e) const {
      return false;
    }

    bool has_var_vis::operator()(const int_literal& e) const {
      return false;
    }

    bool has_var_vis::operator()(const double_literal& e) const {
      return false;
    }

    bool has_var_vis::operator()(const array_expr& e) const {
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      return false;
    }

    bool has_var_vis::operator()(const matrix_expr& e) const {
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      return false;
    }

    bool has_var_vis::operator()(const row_vector_expr& e) const {
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      return false;
    }

    bool has_var_vis::operator()(const variable& e) const {
      scope var_scope = var_map_.get_scope(e.name_);
      return var_scope.par_or_tpar()
        || (var_scope.is_local() && e.type_.base_type_ != INT_T);
    }

    bool has_var_vis::operator()(const fun& e) const {
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      return false;
    }

    bool has_var_vis::operator()(const integrate_ode& e) const {
      // only init state and params may contain vars
      return boost::apply_visitor(*this, e.y0_.expr_)
        || boost::apply_visitor(*this, e.theta_.expr_);
    }

    bool has_var_vis::operator()(const integrate_ode_control& e) const {
      // only init state and params may contain vars
      return boost::apply_visitor(*this, e.y0_.expr_)
        || boost::apply_visitor(*this, e.theta_.expr_);
    }

    bool has_var_vis::operator()(const index_op& e) const {
      return boost::apply_visitor(*this, e.expr_.expr_);
    }

    bool has_var_vis::operator()(const index_op_sliced& e) const {
      return boost::apply_visitor(*this, e.expr_.expr_);
    }

    bool has_var_vis::operator()(const conditional_op& e) const {
      return boost::apply_visitor(*this, e.cond_.expr_)
        || boost::apply_visitor(*this, e.true_val_.expr_)
        || boost::apply_visitor(*this, e.false_val_.expr_);
    }

    bool has_var_vis::operator()(const binary_op& e) const {
      return boost::apply_visitor(*this, e.left.expr_)
        || boost::apply_visitor(*this, e.right.expr_);
    }

    bool has_var_vis::operator()(const unary_op& e) const {
        return boost::apply_visitor(*this, e.subject.expr_);
    }

  }
}
#endif
