#ifndef STAN_LANG_GRAMMARS_STATEMENT_GRAMMAR_DEF_HPP
#define STAN_LANG_GRAMMARS_STATEMENT_GRAMMAR_DEF_HPP

#include <boost/spirit/include/qi.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/recursive_variant.hpp>

#include <boost/spirit/include/version.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/common_adaptors_def.hpp>
#include <stan/lang/grammars/expression_grammar.hpp>
#include <stan/lang/grammars/statement_grammar.hpp>
#include <stan/lang/grammars/var_decls_grammar.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>

#include <cstddef>
#include <iomanip>
#include <istream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>


BOOST_FUSION_ADAPT_STRUCT(stan::lang::assgn,
                          (stan::lang::variable, lhs_var_)
                          (std::vector<stan::lang::idx>, idxs_)
                          (stan::lang::expression, rhs_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::assignment,
                          (stan::lang::variable_dims, var_dims_)
                          (stan::lang::expression, expr_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::variable_dims,
                          (std::string, name_)
                          (std::vector<stan::lang::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::distribution,
                          (std::string, family_)
                          (std::vector<stan::lang::expression>, args_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::for_statement,
                          (std::string, variable_)
                          (stan::lang::range, range_)
                          (stan::lang::statement, statement_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::return_statement,
                          (stan::lang::expression, return_value_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::print_statement,
                          (std::vector<stan::lang::printable>, printables_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::reject_statement,
                          (std::vector<stan::lang::printable>, printables_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::increment_log_prob_statement,
                          (stan::lang::expression, log_prob_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::sample,
                          (stan::lang::expression, expr_)
                          (stan::lang::distribution, dist_)
                          (stan::lang::range, truncation_) )

BOOST_FUSION_ADAPT_STRUCT(stan::lang::statements,
                          (std::vector<stan::lang::var_decl>, local_decl_)
                          (std::vector<stan::lang::statement>, statements_) )

namespace stan {
  namespace lang {

    // see bare_type_grammar_def.hpp for original
    struct set_val4 {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2>
      struct result<F(T1, T2)> { typedef void type; };
      template <typename T1, typename T2>
      void operator()(T1& lhs,
                      const T2& rhs) const {
        lhs = rhs;
      }
    };
    boost::phoenix::function<set_val4> set_val4_f;

    struct validate_return_allowed {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3>
      struct result<F(T1, T2, T3)> { typedef void type; };
      void operator()(var_origin origin,
                      bool& pass,
                      std::ostream& error_msgs) const {
        if (origin != function_argument_origin
            && origin != function_argument_origin_lp
            && origin != function_argument_origin_rng) {
          error_msgs << "Returns only allowed from function bodies."
                     << std::endl;
          pass = false;
          return;
        }
        pass = true;
      }
    };
    boost::phoenix::function<validate_return_allowed>
    validate_return_allowed_f;

    struct validate_void_return_allowed {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3>
      struct result<F(T1, T2, T3)> { typedef void type; };
      void operator()(var_origin origin,
                      bool& pass,
                      std::ostream& error_msgs) const {
        if (origin != void_function_argument_origin
            && origin != void_function_argument_origin_lp
            && origin != void_function_argument_origin_rng) {
          error_msgs << "Void returns only allowed from function"
                     << " bodies of void return type."
                     << std::endl;
          pass = false;
          return;
        }
        pass = true;
      }
    };
    boost::phoenix::function<validate_void_return_allowed>
    validate_void_return_allowed_f;

    struct identifier_to_var {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3,
                typename T4, typename T5, typename T6>
      struct result<F(T1, T2, T3, T4, T5, T6)> { typedef void type; };

      void operator()(const std::string& name,
                      const var_origin& origin_allowed,
                      variable& v,
                      bool& pass,
                      const variable_map& vm,
                      std::ostream& error_msgs) const {
        // validate existence
        if (!vm.exists(name)) {
          error_msgs << "unknown variable in assignment"
                     << "; lhs variable=" << name
                     << std::endl;

          pass = false;
          return;
        }
        // validate origin
        var_origin lhs_origin = vm.get_origin(name);
        if (lhs_origin != local_origin
            && lhs_origin != origin_allowed) {
          error_msgs << "attempt to assign variable in block other than that"
                     << " in which it was defined;"
                     << " left-hand-side variable was declared in block=";
          print_var_origin(error_msgs, lhs_origin);
          error_msgs << " but current top-level block=";
          print_var_origin(error_msgs, origin_allowed);
          error_msgs << std::endl;
          pass = false;
          return;
        }
        // enforce constancy of function args
        if (lhs_origin == function_argument_origin
            || lhs_origin == function_argument_origin_lp
            || lhs_origin == function_argument_origin_rng
            || lhs_origin == void_function_argument_origin
            || lhs_origin == void_function_argument_origin_lp
            || lhs_origin == void_function_argument_origin_rng) {
          error_msgs << "Illegal to assign to function argument variables."
                     << "; use local variables instead."
                     << std::endl;
          pass = false;
          return;
        }

        v = variable(name);
        v.set_type(vm.get_base_type(name), vm.get_num_dims(name));
        pass = true;
      }
    };
    boost::phoenix::function<identifier_to_var> identifier_to_var_f;


    struct validate_assgn {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3>
      struct result<F(T1, T2, T3)> { typedef void type; };

      void operator()(const assgn& a,
                      bool& pass,
                      std::ostream& error_msgs) const {
        // resolve type of lhs[idxs] and make sure it matches rhs
        std::string name = a.lhs_var_.name_;
        expression lhs_expr = expression(a.lhs_var_);
        expr_type lhs_type = indexed_type(lhs_expr, a.idxs_);
        if (lhs_type.is_ill_formed()) {
          pass = false;
          return;
        }

        expr_type rhs_type = a.rhs_.expression_type();
        base_expr_type lhs_base_type = lhs_type.base_type_;
        base_expr_type rhs_base_type = rhs_type.base_type_;
        // allow int -> double promotion, even in arrays
        bool types_compatible
          = lhs_base_type == rhs_base_type
          || (lhs_base_type == DOUBLE_T && rhs_base_type == INT_T);
        if (!types_compatible) {
          error_msgs << "base type mismatch in assignment"
                     << "; variable name="
                     << name
                     << ", type=";
          write_base_expr_type(error_msgs, lhs_base_type);
          error_msgs << "; right-hand side type=";
          write_base_expr_type(error_msgs, rhs_base_type);
          error_msgs << std::endl;
          pass = false;
          return;
        }

        if (lhs_type.num_dims_ != rhs_type.num_dims_) {
          error_msgs << "dimension mismatch in assignment"
                     << "; variable name="
                     << name
                     << ", num dimensions given="
                     << lhs_type.num_dims_
                     << "; right-hand side dimensions="
                     << rhs_type.num_dims_
                     << std::endl;
          pass = false;
          return;
        }

        if (a.lhs_var_occurs_on_rhs()) {
          // this only requires a warning --- a deep copy will be made
          error_msgs << "WARNING: left-hand side variable"
                     << " (name=" << name << ")"
                     << " occurs on right-hand side of assignment, causing"
                     << " inefficient deep copy to avoid aliasing."
                     << std::endl;
        }

        pass = true;
      }
    };
    boost::phoenix::function<validate_assgn> validate_assgn_f;

    struct validate_assignment {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3,
                typename T4, typename T5>
      struct result<F(T1, T2, T3, T4, T5)> { typedef void type; };

      void operator()(assignment& a,
                      const var_origin& origin_allowed,
                      bool& pass,
                      variable_map& vm,
                      std::ostream& error_msgs) const {
        // validate existence
        std::string name = a.var_dims_.name_;
        if (!vm.exists(name)) {
          error_msgs << "unknown variable in assignment"
                     << "; lhs variable=" << a.var_dims_.name_
                     << std::endl;

          pass = false;
          return;
        }

        // validate origin
        var_origin lhs_origin = vm.get_origin(name);
        if (lhs_origin != local_origin
            && lhs_origin != origin_allowed) {
          error_msgs << "attempt to assign variable in wrong block."
                     << " left-hand-side variable origin=";
          print_var_origin(error_msgs, lhs_origin);
          error_msgs << std::endl;
          pass = false;
          return;
        }

        // enforce constancy of function args
        if (lhs_origin == function_argument_origin
            || lhs_origin == function_argument_origin_lp
            || lhs_origin == function_argument_origin_rng
            || lhs_origin == void_function_argument_origin
            || lhs_origin == void_function_argument_origin_lp
            || lhs_origin == void_function_argument_origin_rng) {
          error_msgs << "Illegal to assign to function argument variables."
                     << std::endl
                     << "Use local variables instead."
                     << std::endl;
          pass = false;
          return;
        }


        // validate types
        a.var_type_ = vm.get(name);
        size_t lhs_var_num_dims = a.var_type_.dims_.size();
        size_t num_index_dims = a.var_dims_.dims_.size();

        expr_type lhs_type = infer_type_indexing(a.var_type_.base_type_,
                                                 lhs_var_num_dims,
                                                 num_index_dims);

        if (lhs_type.is_ill_formed()) {
          error_msgs << "too many indexes for variable "
                     << "; variable name = " << name
                     << "; num dimensions given = " << num_index_dims
                     << "; variable array dimensions = " << lhs_var_num_dims
                     << std::endl;
          pass = false;
          return;
        }

        base_expr_type lhs_base_type = lhs_type.base_type_;
        base_expr_type rhs_base_type = a.expr_.expression_type().base_type_;
        // allow int -> double promotion
        bool types_compatible
          = lhs_base_type == rhs_base_type
          || (lhs_base_type == DOUBLE_T && rhs_base_type == INT_T);
        if (!types_compatible) {
          error_msgs << "base type mismatch in assignment"
                     << "; variable name = "
                     << a.var_dims_.name_
                     << ", type = ";
          write_base_expr_type(error_msgs, lhs_base_type);
          error_msgs << "; right-hand side type=";
          write_base_expr_type(error_msgs, rhs_base_type);
          error_msgs << std::endl;
          pass = false;
          return;
        }
        if (lhs_type.num_dims_ != a.expr_.expression_type().num_dims_) {
          error_msgs << "dimension mismatch in assignment"
                     << "; variable name = "
                     << a.var_dims_.name_
                     << ", num dimensions given = "
                     << lhs_type.num_dims_
                     << "; right-hand side dimensions = "
                     << a.expr_.expression_type().num_dims_
                     << std::endl;
          pass = false;
          return;
        }
        
        pass = true;
      }
    };
    boost::phoenix::function<validate_assignment> validate_assignment_f;

    struct validate_sample {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3, typename T4>
      struct result<F(T1, T2, T3, T4)> { typedef void type; };

      bool is_double_return(const std::string& function_name,
                            const std::vector<expr_type>& arg_types,
                            std::ostream& error_msgs) const {
        return function_signatures::instance()
          .get_result_type(function_name, arg_types, error_msgs, true)
          .is_primitive_double();
      }
      static bool is_univariate(const expr_type& et) {
        return et.num_dims_ == 0
          && (et.base_type_ == INT_T
              || et.base_type_ == DOUBLE_T);
      }
      void operator()(const sample& s,
                      const variable_map& var_map,
                      bool& pass,
                      std::ostream& error_msgs) const {
        static const bool user_facing = true;
        std::vector<expr_type> arg_types;
        arg_types.push_back(s.expr_.expression_type());
        for (size_t i = 0; i < s.dist_.args_.size(); ++i)
          arg_types.push_back(s.dist_.args_[i].expression_type());
        std::string function_name(s.dist_.family_);
        std::string internal_function_name = function_name + "_log";

        if ((internal_function_name.find("multiply_log")
             != std::string::npos)
            || (internal_function_name.find("binomial_coefficient_log")
                != std::string::npos)) {
          error_msgs << "Only distribution names can be used with"
                     << " sampling (~) notation; found non-distribution"
                     << " function: " << function_name
                     << std::endl;
          pass = false;
          return;
        }

        if (internal_function_name.find("cdf_log") != std::string::npos) {
          error_msgs << "CDF and CCDF functions may not be used with"
                     << " sampling notation."
                     << " Use increment_log_prob("
                     << internal_function_name << "(...)) instead."
                     << std::endl;
          pass = false;
          return;
        }

        if (!is_double_return(internal_function_name, arg_types, error_msgs)) {
          pass = false;
          return;
        }

        if (internal_function_name == "lkj_cov_log") {
          error_msgs << "Warning: the lkj_cov_log() sampling distribution"
                     << " is deprecated.  It will be removed in Stan 3."
                     << std::endl
                     << "Code LKJ covariance in terms of an lkj_corr()"
                     << " distribution on a correlation matrix"
                     << " and independent lognormals on the scales."
                     << std::endl << std::endl;
        }

        // test for LHS not being purely a variable
        if (has_non_param_var(s.expr_, var_map)) {
          // TODO(carpenter):  really want to get line numbers in here too
          error_msgs << "Warning (non-fatal):"
             << std::endl
             << "Left-hand side of sampling statement (~) may contain a"
             << " non-linear transform of a parameter or local variable."
             << std::endl
             << "If so, you need to call increment_log_prob() with the log"
             << " absolute determinant of the Jacobian of the transform."
             << std::endl
             << "Left-hand-side of sampling statement:"
             << std::endl
             << "    ";
          generate_expression(s.expr_, user_facing, error_msgs);
          error_msgs << " ~ " << function_name << "(...)"
                     << std::endl;
        }
        // validate that variable and params are univariate if truncated
        if (s.truncation_.has_low() || s.truncation_.has_high()) {
          if (!is_univariate(s.expr_.expression_type())) {
            error_msgs << "Outcomes in truncated distributions"
                       << " must be univariate."
                       << std::endl
                       << "  Found outcome expression: ";
            generate_expression(s.expr_, user_facing, error_msgs);
            error_msgs << std::endl
                       << "  with non-univariate type: "
                       << s.expr_.expression_type()
                       << std::endl;
            pass = false;
            return;
          }
          for (size_t i = 0; i < s.dist_.args_.size(); ++i)
            if (!is_univariate(s.dist_.args_[i].expression_type())) {
              error_msgs << "Parameters in truncated distributions"
                         << " must be univariate."
                         << std::endl
                         << "  Found parameter expression: ";
              generate_expression(s.dist_.args_[i], user_facing, error_msgs);
              error_msgs << std::endl
                         << "  with non-univariate type: "
                         << s.dist_.args_[i].expression_type()
                         << std::endl;
              pass = false;
              return;
            }
        }
        if (s.truncation_.has_low()
            && !is_univariate(s.truncation_.low_.expression_type())) {
          error_msgs << "Lower bounds in truncated distributions"
                     << " must be univariate."
                     << std::endl
                     << "  Found lower bound expression: ";
          generate_expression(s.truncation_.low_, user_facing, error_msgs);
          error_msgs << std::endl
                     << "  with non-univariate type: "
                     << s.truncation_.low_.expression_type()
                     << std::endl;
          pass = false;
          return;
        }
        if (s.truncation_.has_high()
            && !is_univariate(s.truncation_.high_.expression_type())) {
          error_msgs << "Upper bounds in truncated distributions"
                     << " must be univariate."
                     << std::endl
                     << "  Found upper bound expression: ";
          generate_expression(s.truncation_.high_, user_facing, error_msgs);
          error_msgs << std::endl
                     << "  with non-univariate type: "
                     << s.truncation_.high_.expression_type()
                     << std::endl;
          pass = false;
          return;
        }

        if (s.truncation_.has_low()) {
          std::vector<expr_type> arg_types_trunc(arg_types);
          arg_types_trunc[0] = s.truncation_.low_.expression_type();
          std::string function_name_cdf(s.dist_.family_);
          function_name_cdf += "_cdf_log";
          if (!is_double_return(function_name_cdf, arg_types_trunc,
                                error_msgs)) {
            error_msgs << "lower truncation not defined for specified"
                       << " arguments to "
                       << s.dist_.family_ << std::endl;
            pass = false;
            return;
          }
          if (!is_double_return(function_name_cdf, arg_types, error_msgs)) {
            error_msgs << "lower bound in truncation type does not match"
                       << " sampled variate in distribution's type"
                       << std::endl;
            pass = false;
            return;
          }
        }
        if (s.truncation_.has_high()) {
          std::vector<expr_type> arg_types_trunc(arg_types);
          arg_types_trunc[0] = s.truncation_.high_.expression_type();
          std::string function_name_cdf(s.dist_.family_);
          function_name_cdf += "_cdf_log";
          if (!is_double_return(function_name_cdf, arg_types_trunc,
                                error_msgs)) {
            error_msgs << "upper truncation not defined for"
                       << " specified arguments to "
                       << s.dist_.family_ << std::endl;

            pass = false;
            return;
          }
          if (!is_double_return(function_name_cdf, arg_types, error_msgs)) {
            error_msgs << "upper bound in truncation type does not match"
                       << " sampled variate in distribution's type"
                       << std::endl;
            pass = false;
            return;
          }
        }
        pass = true;
      }
    };
    boost::phoenix::function<validate_sample> validate_sample_f;

    struct expression_as_statement {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3>
      struct result<F(T1, T2, T3)> { typedef void type; };
      void operator()(bool& pass,
                      const stan::lang::expression& expr,
                      std::stringstream& error_msgs) const {
        static const bool user_facing = true;
        if (expr.expression_type() != VOID_T) {
          error_msgs << "Illegal statement beginning with non-void"
                     << " expression parsed as"
                     << std::endl << "  ";
          generate_expression(expr.expr_, user_facing, error_msgs);
          error_msgs << std::endl
                     << "Not a legal assignment, sampling, or function"
                     << " statement.  Note that"
                     << std::endl
                     << "  * Assignment statements only allow variables"
                     << " (with optional indexes) on the left;"
                     << std::endl
                     << "    if you see an outer function logical_lt (<)"
                     << " with negated (-) second argument,"
                     << std::endl
                     << "    it indicates an assignment statement A <- B"
                     << " with illegal left"
                     << std::endl
                     << "    side A parsed as expression (A < (-B))."
                     << std::endl
                     << "  * Sampling statements allow arbitrary"
                     << " value-denoting expressions on the left."
                     << std::endl
                     << "  * Functions used as statements must be"
                     << " declared to have void returns"
                     << std::endl << std::endl;
          pass = false;
          return;
        }
        pass = true;
      }
    };
    boost::phoenix::function<expression_as_statement> expression_as_statement_f;

    struct unscope_locals {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2>
      struct result<F(T1, T2)> { typedef void type; };
      void operator()(const std::vector<var_decl>& var_decls,
                      variable_map& vm) const {
        for (size_t i = 0; i < var_decls.size(); ++i)
          vm.remove(var_decls[i].name());
      }
    };
    boost::phoenix::function<unscope_locals> unscope_locals_f;

    struct add_while_condition {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3, typename T4>
      struct result<F(T1, T2, T3, T4)> { typedef void type; };
      void operator()(while_statement& ws,
                      const expression& e,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        pass = e.expression_type().is_primitive();
        if (!pass) {
          error_msgs << "conditions in while statement must be primitive"
                     << " int or real;"
                     << " found type=" << e.expression_type() << std::endl;
          return;
        }
        ws.condition_ = e;
      }
    };
    boost::phoenix::function<add_while_condition> add_while_condition_f;

    struct add_while_body {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2>
      struct result<F(T1, T2)> { typedef void type; };
      void operator()(while_statement& ws,
                      const statement& s) const {
        ws.body_ = s;
      }
    };
    boost::phoenix::function<add_while_body> add_while_body_f;

    struct add_loop_identifier {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3, typename T4,
                typename T5>
      struct result<F(T1, T2, T3, T4, T5)> { typedef void type; };
      void operator()(const std::string& name,
                      std::string& name_local,
                      bool& pass,
                      variable_map& vm,
                      std::stringstream& error_msgs) const {
        name_local = name;
        pass = !vm.exists(name);
        if (!pass) {
          error_msgs << "ERROR: loop variable already declared."
                     << " variable name=\"" << name << "\"" << std::endl;
        } else {
          vm.add(name, base_var_decl(name, std::vector<expression>(), INT_T),
                 local_origin);  // loop var acts like local
        }
      }
    };
    boost::phoenix::function<add_loop_identifier> add_loop_identifier_f;

    struct remove_loop_identifier {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2>
      struct result<F(T1, T2)> { typedef void type; };
      void operator()(const std::string& name,
                      variable_map& vm) const {
        vm.remove(name);
      }
    };
    boost::phoenix::function<remove_loop_identifier> remove_loop_identifier_f;

    struct validate_int_expr2 {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3>
      struct result<F(T1, T2, T3)> { typedef void type; };

      void operator()(const expression& expr,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        if (!expr.expression_type().is_primitive_int()) {
          error_msgs << "expression denoting integer required; found type="
                     << expr.expression_type() << std::endl;
          pass = false;
          return;
        }
        pass = true;
        return;
      }
    };
    boost::phoenix::function<validate_int_expr2> validate_int_expr2_f;

    struct validate_allow_sample {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3>
      struct result<F(T1, T2, T3)> { typedef void type; };

      void operator()(const bool& allow_sample,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        if (!allow_sample) {
          error_msgs << "Sampling statements (~) and increment_log_prob() are"
                     << std::endl
                     << "only allowed in the model block."
                     << std::endl;
          pass = false;
          return;
        }
        pass = true;
        return;
      }
    };
    boost::phoenix::function<validate_allow_sample> validate_allow_sample_f;

    struct validate_non_void_expression {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3>
      struct result<F(T1, T2, T3)> { typedef void type; };

      void operator()(const expression& e,
                      bool& pass,
                      std::ostream& error_msgs) const {
        pass = !e.expression_type().is_void();
        if (!pass) {
          error_msgs << "attempt to increment log prob with void expression"
                     << std::endl;
        }
      }
    };
    boost::phoenix::function<validate_non_void_expression>
    validate_non_void_expression_f;

    struct add_line_number {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1, typename T2, typename T3>
      struct result<F(T1, T2, T3)> { typedef void type; };
      template <typename T, typename It>
      void operator()(T& stmt,
                      It& begin,
                      It& end) const {
        stmt.begin_line_ = get_line(begin);
        stmt.end_line_ = get_line(end);
      }
    };
    boost::phoenix::function<add_line_number> add_line_number_f;

    struct set_void_return {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1>
      struct result<F(T1)> { typedef void type; };
      void operator()(return_statement& s) const {
        s = return_statement();
      }
    };
    boost::phoenix::function<set_void_return> set_void_return_f;

    struct set_no_op {
      //! @cond Doxygen_Suppress
      template <class> struct result;
      //! @endcond
      template <typename F, typename T1>
      struct result<F(T1)> { typedef void type; };
      void operator()(no_op_statement& s) const {
        s = no_op_statement();
      }
    };
    boost::phoenix::function<set_no_op> set_no_op_f;

    template <typename Iterator>
    statement_grammar<Iterator>::statement_grammar(variable_map& var_map,
                                           std::stringstream& error_msgs)
      : statement_grammar::base_type(statement_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        expression_g(var_map, error_msgs),
        var_decls_g(var_map, error_msgs),
        statement_2_g(var_map, error_msgs, *this),
        indexes_g(var_map, error_msgs, expression_g) {
      using boost::spirit::qi::_1;
      using boost::spirit::qi::char_;
      using boost::spirit::qi::eps;
      using boost::spirit::qi::lexeme;
      using boost::spirit::qi::lit;
      using boost::spirit::qi::no_skip;
      using boost::spirit::qi::_pass;
      using boost::spirit::qi::_val;
      using boost::spirit::qi::raw;

      using boost::spirit::qi::labels::_a;
      using boost::spirit::qi::labels::_r1;
      using boost::spirit::qi::labels::_r2;
      using boost::spirit::qi::labels::_r3;

      using boost::phoenix::begin;
      using boost::phoenix::end;

      // inherited features
      //   _r1 true if sample_r allowed
      //   _r2 source of variables allowed for assignments
      //   _r3 true if return_r allowed

      // raw[ ] just to wrap to get line numbers
      statement_r.name("statement");
      statement_r
        = raw[statement_sub_r(_r1, _r2, _r3)[set_val4_f(_val, _1)]]
        [add_line_number_f(_val, begin(_1), end(_1))];

      statement_sub_r.name("statement");
      statement_sub_r
        %= no_op_statement_r                        // key ";"
        | statement_seq_r(_r1, _r2, _r3)              // key "{"
        | increment_log_prob_statement_r(_r1, _r2)  // key "increment_log_prob"
        | for_statement_r(_r1, _r2, _r3)              // key "for"
        | while_statement_r(_r1, _r2, _r3)            // key "while"
        | statement_2_g(_r1, _r2, _r3)                // key "if"
        | print_statement_r(_r2)                    // key "print"
        | reject_statement_r(_r2)                   // key "reject"
        | return_statement_r(_r2)                   // key "return"
        | void_return_statement_r(_r2)              // key "return"
        | assignment_r(_r2)                         // lvalue "<-"
        | assgn_r(_r2)                              // var[idxs] <- expr
        | sample_r(_r1, _r2)                         // expression "~"
        | expression_g(_r2)                         // expression
        [expression_as_statement_f(_pass, _1,
                                   boost::phoenix::ref(error_msgs_))];

      // _r1, _r2, _r3 same as statement_r
      statement_seq_r.name("sequence of statements");
      statement_seq_r
        %= lit('{')
        > local_var_decls_r[set_val4_f(_a, _1)]
        > *statement_r(_r1, _r2, _r3)
        > lit('}')
        > eps[unscope_locals_f(_a, boost::phoenix::ref(var_map_))];

      local_var_decls_r
        %= var_decls_g(false, local_origin);  // - constants

      // inherited  _r1 = true if samples allowed as statements
      increment_log_prob_statement_r.name("increment log prob statement");
      increment_log_prob_statement_r
        %= (lit("increment_log_prob") >> no_skip[!char_("a-zA-Z0-9_")])
        > eps[ validate_allow_sample_f(_r1, _pass,
                                       boost::phoenix::ref(error_msgs_)) ]
        > lit('(')
        > expression_g(_r2)
          [validate_non_void_expression_f(_1, _pass,
                                          boost::phoenix::ref(error_msgs_))]
        > lit(')')
        > lit(';');

      // _r1, _r2, _r3 same as statement_r
      while_statement_r.name("while statement");
      while_statement_r
        = (lit("while") >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('(')
        > expression_g(_r2)
          [add_while_condition_f(_val, _1, _pass,
                                 boost::phoenix::ref(error_msgs_))]
        > lit(')')
        > statement_r(_r1, _r2, _r3)
          [add_while_body_f(_val, _1)];


      // _r1, _r2, _r3 same as statement_r
      for_statement_r.name("for statement");
      for_statement_r
        %= (lit("for") >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('(')
        > identifier_r[add_loop_identifier_f(_1, _a, _pass,
                                         boost::phoenix::ref(var_map_),
                                         boost::phoenix::ref(error_msgs_))]
        > lit("in")
        > range_r(_r2)
        > lit(')')
        > statement_r(_r1, _r2, _r3)
        > eps
        [remove_loop_identifier_f(_a, boost::phoenix::ref(var_map_))];

      print_statement_r.name("print statement");
      print_statement_r
        %= (lit("print") >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('(')
        > (printable_r(_r1) % ',')
        > lit(')');

      // reject
      reject_statement_r.name("reject statement");
      reject_statement_r
        %= (lit("reject") >> no_skip[!char_("a-zA-Z0-9_")])
        > lit('(')
        > (printable_r(_r1) % ',')
        > lit(')');

      printable_r.name("printable");
      printable_r
        %= printable_string_r
        | expression_g(_r1);

      printable_string_r.name("printable quoted string");
      printable_string_r
        %= lit('"')
        > no_skip[*char_("a-zA-Z0-9/~!@#$%^&*()`_+-={}|[]:;'<>?,./ ")]
        > lit('"');

      identifier_r.name("identifier");
      identifier_r
        %= (lexeme[char_("a-zA-Z")
                   >> *char_("a-zA-Z0-9_.")]);

      range_r.name("range expression pair, colon");
      range_r
        %= expression_g(_r1)
           [validate_int_expr2_f(_1, _pass, boost::phoenix::ref(error_msgs_))]
        >> lit(':')
        >> expression_g(_r1)
           [validate_int_expr2_f(_1, _pass, boost::phoenix::ref(error_msgs_))];

      assignment_r.name("variable assignment by expression");
      assignment_r
        %= (var_lhs_r(_r1)
            >> lit("<-"))
        > expression_rhs_r(_r1)
          [validate_assignment_f(_val, _r1, _pass,
                                 boost::phoenix::ref(var_map_),
                                 boost::phoenix::ref(error_msgs_))]
        > lit(';');

      assgn_r.name("assginment statement");
      assgn_r
        %= var_r(_r1)
        >> indexes_g(_r1)
        >> lit("<-")
        >> (eps > expression_rhs_r(_r1))
           [validate_assgn_f(_val, _pass, boost::phoenix::ref(error_msgs_))]
        > lit(';');

      var_r.name("variable for left-hand side of assignment");
      var_r
        = identifier_r
          [identifier_to_var_f(_1, _r1, _val,  _pass,
                               boost::phoenix::ref(var_map_),
                               boost::phoenix::ref(error_msgs_))];

      expression_rhs_r.name("expression assignable to left-hand side");
      expression_rhs_r
        %= expression_g(_r1);

      var_lhs_r.name("variable and array dimensions");
      var_lhs_r
        %= identifier_r
        >> opt_dims_r(_r1);

      opt_dims_r.name("array dimensions (optional)");
      opt_dims_r
        %=  *dims_r(_r1);

      dims_r.name("array dimensions");
      dims_r
        %= lit('[')
        >> (expression_g(_r1)
           [validate_int_expr2_f(_1, _pass, boost::phoenix::ref(error_msgs_))]
            % ',')
        >> lit(']');

      // inherited  _r1 = true if samples allowed as statements
      sample_r.name("distribution of expression");
      sample_r
        %= (expression_g(_r2)
            >> lit('~'))
        > eps
          [validate_allow_sample_f(_r1, _pass,
                                   boost::phoenix::ref(error_msgs_))]
        > distribution_r(_r2)
        > -truncation_range_r(_r2)
        > lit(';')
        > eps
          [validate_sample_f(_val, boost::phoenix::ref(var_map_),
                             _pass, boost::phoenix::ref(error_msgs_))];

      distribution_r.name("distribution and parameters");
      distribution_r
        %= (identifier_r
            >> lit('(')
            >> -(expression_g(_r1) % ','))
        > lit(')');

      truncation_range_r.name("range pair");
      truncation_range_r
        %= lit('T')
        > lit('[')
        > -expression_g(_r1)
        > lit(',')
        > -expression_g(_r1)
        > lit(']');

      // _r1 = allow sampling, _r2 = var origin
      return_statement_r.name("return statement");
      return_statement_r
        %= (lit("return") >> no_skip[!char_("a-zA-Z0-9_")])
        >> expression_g(_r1)
        >> lit(';') [validate_return_allowed_f(_r1, _pass,
                                       boost::phoenix::ref(error_msgs_))];

      // _r1 = var origin
      void_return_statement_r.name("void return statement");
      void_return_statement_r
        = lit("return")[set_void_return_f(_val)]  // = expression()]
        >> lit(';')[validate_void_return_allowed_f(_r1, _pass,
                                        boost::phoenix::ref(error_msgs_))];

      no_op_statement_r.name("no op statement");
      no_op_statement_r
        %= lit(';')[set_no_op_f(_val)];
    }

  }
}
#endif
