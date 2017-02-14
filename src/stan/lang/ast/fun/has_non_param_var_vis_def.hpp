#ifndef STAN_LANG_AST_FUN_HAS_NON_PARAM_VAR_VIS_DEF_HPP
#define STAN_LANG_AST_FUN_HAS_NON_PARAM_VAR_VIS_DEF_HPP

#include <stan/lang/ast.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <string>

namespace stan {
  namespace lang {

    bool is_linear_function(const std::string& name) {
      return name == "add"
        || name == "block"
        || name == "append_col"
        || name == "col"
        || name == "cols"
        || name == "diagonal"
        || name == "head"
        || name == "minus"
        || name == "negative_infinity"
        || name == "not_a_number"
        || name == "append_row"
        || name == "rep_matrix"
        || name == "rep_row_vector"
        || name == "rep_vector"
        || name == "row"
        || name == "rows"
        || name == "positive_infinity"
        || name == "segment"
        || name == "subtract"
        || name == "sum"
        || name == "tail"
        || name == "to_vector"
        || name == "to_row_vector"
        || name == "to_matrix"
        || name == "to_array_1d"
        || name == "to_array_2d"
        || name == "transpose";
    }

    has_non_param_var_vis::has_non_param_var_vis(const variable_map& var_map)
      : var_map_(var_map) {
    }

    bool has_non_param_var_vis::operator()(const nil& e) const {
      return false;
    }

    bool has_non_param_var_vis::operator()(const int_literal& e) const {
      return false;
    }

    bool has_non_param_var_vis::operator()(const double_literal& e) const {
      return false;
    }

    bool has_non_param_var_vis::operator()(const array_expr& e) const {
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      return false;
    }

    bool has_non_param_var_vis::operator()(const variable& e) const {
      var_origin vo = var_map_.get_origin(e.name_);
      return vo == transformed_parameter_origin
        || vo == local_origin;
    }

    bool has_non_param_var_vis::operator()(const integrate_ode& e) const {
      // if any vars, return true because integration will be nonlinear
      return boost::apply_visitor(*this, e.y0_.expr_)
        || boost::apply_visitor(*this, e.theta_.expr_);
    }

    bool has_non_param_var_vis::operator()(const integrate_ode_control& e)
      const {
      // if any vars, return true because integration will be nonlinear
      return boost::apply_visitor(*this, e.y0_.expr_)
        || boost::apply_visitor(*this, e.theta_.expr_);
    }

    bool has_non_param_var_vis::operator()(const generalOdeModel_control& e)
      const {
      // CHECK - anything to do with nonlinearity ?
      // Putting in the variables that may contain var types
      return ((((((boost::apply_visitor(*this, e.time_.expr_)
        || boost::apply_visitor(*this, e.amt_.expr_))
        || boost::apply_visitor(*this, e.rate_.expr_))
        || boost::apply_visitor(*this, e.ii_.expr_))
        || boost::apply_visitor(*this, e.pMatrix_.expr_))
        || boost::apply_visitor(*this, e.biovar_.expr_))
        || boost::apply_visitor(*this, e.tlag_.expr_));
    }

    bool has_non_param_var_vis::operator()(const fun& e) const {
      // any function applied to non-linearly transformed var
      for (size_t i = 0; i < e.args_.size(); ++i)
        if (boost::apply_visitor(*this, e.args_[i].expr_))
          return true;
      // non-linear function applied to var
      if (!is_linear_function(e.name_)) {
        for (size_t i = 0; i < e.args_.size(); ++i)
          if (has_var(e.args_[i], var_map_))
            return true;
      }
      return false;
    }

    bool has_non_param_var_vis::operator()(const index_op& e) const {
      return boost::apply_visitor(*this, e.expr_.expr_);
    }

    bool has_non_param_var_vis::operator()(const index_op_sliced& e) const {
      return boost::apply_visitor(*this, e.expr_.expr_);
    }

    bool has_non_param_var_vis::operator()(const conditional_op& e) const {
      if (has_non_param_var(e.cond_, var_map_)
          || has_non_param_var(e.true_val_, var_map_)
          || has_non_param_var(e.false_val_, var_map_))
        return true;
      return false;
    }

    bool has_non_param_var_vis::operator()(const binary_op& e) const {
      if (e.op == "||"
          || e.op == "&&"
          || e.op == "=="
          || e.op == "!="
          || e.op == "<"
          || e.op == "<="
          || e.op == ">"
          || e.op == ">=")
        return true;
      if (has_non_param_var(e.left, var_map_)
          || has_non_param_var(e.right, var_map_))
        return true;
      if (e.op == "*" || e.op == "/")
        return has_var(e.left, var_map_) && has_var(e.right, var_map_);
      return false;
    }

    bool has_non_param_var_vis::operator()(const unary_op& e) const {
      // only negation, which is linear, so recurse
      return has_non_param_var(e.subject, var_map_);
    }

  }
}
#endif
