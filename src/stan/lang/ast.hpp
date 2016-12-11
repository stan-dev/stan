#ifndef STAN_LANG_AST_HPP
#define STAN_LANG_AST_HPP

#include <stan/lang/ast/base_expr_type.hpp>
#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/nil.hpp>
#include <stan/lang/ast/printable.hpp>
#include <stan/lang/ast/var_origin.hpp>
#include <stan/lang/ast/variable_map.hpp>

#include <stan/lang/ast/fun/is_multi_index_vis.hpp>
#include <stan/lang/ast/fun/is_nil_vis.hpp>
#include <stan/lang/ast/fun/name_vis.hpp>
#include <stan/lang/ast/fun/var_decl_base_type_vis.hpp>
#include <stan/lang/ast/fun/var_decl_def_vis.hpp>
#include <stan/lang/ast/fun/var_decl_dims_vis.hpp>
#include <stan/lang/ast/fun/var_decl_has_def_vis.hpp>

#include <stan/lang/ast/fun/infer_type_indexing.hpp>
#include <stan/lang/ast/fun/is_data_origin.hpp>
#include <stan/lang/ast/fun/is_fun_origin.hpp>
#include <stan/lang/ast/fun/print_var_origin.hpp>
#include <stan/lang/ast/fun/total_dims.hpp>
#include <stan/lang/ast/fun/write_base_expr_type.hpp>
#include <stan/lang/ast/fun/is_multi_index.hpp>
#include <stan/lang/ast/fun/is_nil.hpp>
#include <stan/lang/ast/fun/operator_stream_expr_type.hpp>
#include <stan/lang/ast/fun/promote_primitive.hpp>

#include <stan/lang/ast/sigs/function_signature_t.hpp>
#include <stan/lang/ast/sigs/function_signatures.hpp>

#include <stan/lang/ast/node/array_expr.hpp>
#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/binary_op.hpp>
#include <stan/lang/ast/node/cholesky_corr_var_decl.hpp>
#include <stan/lang/ast/node/cholesky_factor_var_decl.hpp>
#include <stan/lang/ast/node/conditional_op.hpp>
#include <stan/lang/ast/node/corr_matrix_var_decl.hpp>
#include <stan/lang/ast/node/cov_matrix_var_decl.hpp>
#include <stan/lang/ast/node/distribution.hpp>
#include <stan/lang/ast/node/double_literal.hpp>
#include <stan/lang/ast/node/double_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/expression_type_vis.hpp>
#include <stan/lang/ast/node/fun.hpp>
#include <stan/lang/ast/node/idx.hpp>
#include <stan/lang/ast/node/index_op.hpp>
#include <stan/lang/ast/node/index_op_sliced.hpp>
#include <stan/lang/ast/node/integrate_ode.hpp>
#include <stan/lang/ast/node/integrate_ode_control.hpp>
#include <stan/lang/ast/node/int_literal.hpp>
#include <stan/lang/ast/node/int_var_decl.hpp>
#include <stan/lang/ast/node/lb_idx.hpp>
#include <stan/lang/ast/node/lub_idx.hpp>
#include <stan/lang/ast/node/matrix_var_decl.hpp>
#include <stan/lang/ast/node/multi_idx.hpp>
#include <stan/lang/ast/node/omni_idx.hpp>
#include <stan/lang/ast/node/ordered_var_decl.hpp>
#include <stan/lang/ast/node/positive_ordered_var_decl.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <stan/lang/ast/node/row_vector_var_decl.hpp>
#include <stan/lang/ast/node/simplex_var_decl.hpp>
#include <stan/lang/ast/node/statement.hpp>
#include <stan/lang/ast/node/statements.hpp>
#include <stan/lang/ast/node/ub_idx.hpp>
#include <stan/lang/ast/node/unary_op.hpp>
#include <stan/lang/ast/node/uni_idx.hpp>
#include <stan/lang/ast/node/unit_vector_var_decl.hpp>
#include <stan/lang/ast/node/variable.hpp>
#include <stan/lang/ast/node/variable_dims.hpp>
#include <stan/lang/ast/node/var_decl.hpp>
#include <stan/lang/ast/node/vector_var_decl.hpp>


// ======================================================
#include <boost/variant/recursive_variant.hpp>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace stan {

  namespace lang {

    // components of abstract syntax tree
    struct array_expr;
    struct assignment;
    struct assgn;
    struct binary_op;
    struct break_continue_statement;
    struct conditional_op;
    struct conditional_statement;
    struct distribution;
    struct double_var_decl;
    struct double_literal;
    struct expression;
    struct for_statement;
    struct fun;
    struct function_decl_def;
    struct function_decl_defs;
    struct identifier;
    struct increment_log_prob_statement;
    struct index_op;
    struct index_op_sliced;
    struct int_literal;
    struct inv_var_decl;
    struct matrix_var_decl;
    struct no_op_statement;
    struct ordered_var_decl;
    struct positive_ordered_var_decl;
    struct print_statement;
    struct program;
    struct range;
    struct reject_statement;
    struct return_statement;
    struct row_vector_var_decl;
    struct sample;
    struct simplex_var_decl;
    struct int_var_decl;
    struct integrate_ode;
    struct integrate_ode_control;
    struct unit_vector_var_decl;
    struct statement;
    struct statements;
    struct unary_op;
    struct variable;
    struct variable_dims;
    struct var_decl;
    struct var_type;
    struct vector_var_decl;
    struct while_statement;

     struct is_no_op_statement_vis : public boost::static_visitor<bool> {
      bool operator()(const nil& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const assignment& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const assgn& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const sample& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const increment_log_prob_statement& t) const;  // NOLINT
      bool operator()(const expression& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const statements& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const for_statement& st) const;  // NOLINT
      bool operator()(const conditional_statement& st) const;  // NOLINT
      bool operator()(const while_statement& st) const;  // NOLINT
      bool operator()(const break_continue_statement& st) const;  // NOLINT
      bool operator()(const print_statement& st) const;  // NOLINT
      bool operator()(const reject_statement& st) const;  // NOLINT
      bool operator()(const no_op_statement& st) const;  // NOLINT
      bool operator()(const return_statement& st) const;  // NOLINT
    };


    struct returns_type_vis : public boost::static_visitor<bool> {
      expr_type return_type_;
      std::ostream& error_msgs_;
      returns_type_vis(const expr_type& return_type,
                       std::ostream& error_msgs);
      bool operator()(const nil& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const assignment& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const assgn& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const sample& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const increment_log_prob_statement& t) const;  // NOLINT
      bool operator()(const expression& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const statements& st) const;  // NOLINT(runtime/explicit)
      bool operator()(const for_statement& st) const;  // NOLINT
      bool operator()(const conditional_statement& st) const;  // NOLINT
      bool operator()(const while_statement& st) const;  // NOLINT
      bool operator()(const break_continue_statement& st) const;  // NOLINT
      bool operator()(const print_statement& st) const;  // NOLINT
      bool operator()(const reject_statement& st) const;  // NOLINT
      bool operator()(const no_op_statement& st) const;  // NOLINT
      bool operator()(const return_statement& st) const;  // NOLINT
    };

    bool returns_type(const expr_type& return_type,
                      const statement& statement,
                      std::ostream& error_msgs);

    struct increment_log_prob_statement {
      expression log_prob_;
      increment_log_prob_statement();
      increment_log_prob_statement(const expression& log_prob);  // NOLINT
    };

    struct for_statement {
      std::string variable_;
      range range_;
      statement statement_;
      for_statement();
      for_statement(std::string& variable,
                    range& range,
                    statement& stmt);
    };

    // bodies may be 1 longer than conditions due to else
    struct conditional_statement {
      std::vector<expression> conditions_;
      std::vector<statement> bodies_;
      conditional_statement();
      conditional_statement(const std::vector<expression>& conditions,
                            const std::vector<statement>& statements);
    };

    struct while_statement {
      expression condition_;
      statement body_;
      while_statement();
      while_statement(const expression& condition,
                      const statement& body);
    };

    struct break_continue_statement {
      std::string generate_;
      break_continue_statement();
      explicit break_continue_statement(const std::string& generate);
    };

    struct print_statement {
      std::vector<printable> printables_;
      print_statement();
      print_statement(const std::vector<printable>& printables);  // NOLINT
    };

    struct reject_statement {
      std::vector<printable> printables_;
      reject_statement();
      reject_statement(const std::vector<printable>& printables);  // NOLINT
    };

    struct return_statement {
      expression return_value_;
      return_statement();
      return_statement(const expression& expr);  // NOLINT(runtime/explicit)
    };

    struct no_op_statement {
      // no op, no data
    };

    struct arg_decl {
      expr_type arg_type_;
      std::string name_;
      arg_decl();
      arg_decl(const expr_type& arg_type,
               const std::string& name);
      base_var_decl base_variable_declaration();
    };

    struct function_decl_def {
      function_decl_def();
      function_decl_def(const expr_type& return_type,
                        const std::string& name,
                        const std::vector<arg_decl>& arg_decls,
                        const statement& body);
      expr_type return_type_;
      std::string name_;
      std::vector<arg_decl> arg_decls_;
      statement body_;
    };

    struct function_decl_defs {
      function_decl_defs();

      function_decl_defs(
          const std::vector<function_decl_def>& decl_defs);  // NOLINT

      std::vector<function_decl_def> decl_defs_;
    };


    struct program {
      std::vector<function_decl_def> function_decl_defs_;
      std::vector<var_decl> data_decl_;
      std::pair<std::vector<var_decl>, std::vector<statement> >
      derived_data_decl_;
      std::vector<var_decl> parameter_decl_;
      std::pair<std::vector<var_decl>, std::vector<statement> >
      derived_decl_;
      statement statement_;
      std::pair<std::vector<var_decl>, std::vector<statement> >
      generated_decl_;

      program();
      program(const std::vector<function_decl_def>& function_decl_defs,
              const std::vector<var_decl>& data_decl,
              const std::pair<std::vector<var_decl>,
              std::vector<statement> >& derived_data_decl,
              const std::vector<var_decl>& parameter_decl,
              const std::pair<std::vector<var_decl>,
              std::vector<statement> >& derived_decl,
              const statement& st,
              const std::pair<std::vector<var_decl>,
              std::vector<statement> >& generated_decl);
    };

    struct sample {
      expression expr_;
      distribution dist_;
      range truncation_;
      bool is_discrete_;
      sample();
      sample(expression& e,
             distribution& dist);
      bool is_ill_formed() const;
      bool is_discrete() const;
    };

    struct assignment {
      variable_dims var_dims_;  // lhs_var[dim0,...,dimN-1]
      expression expr_;  // = rhs
      base_var_decl var_type_;  // type of lhs_var
      assignment();
      assignment(variable_dims& var_dims,
                 expression& expr);
    };


    struct var_occurs_vis : public boost::static_visitor<bool> {
      const std::string var_name_;
      explicit var_occurs_vis(const variable& e);
      bool operator()(const nil& e) const;
      bool operator()(const int_literal& e) const;
      bool operator()(const double_literal& e) const;
      bool operator()(const array_expr& e) const;
      bool operator()(const variable& e) const;
      bool operator()(const fun& e) const;
      bool operator()(const integrate_ode& e) const;
      bool operator()(const integrate_ode_control& e) const;
      bool operator()(const index_op& e) const;
      bool operator()(const index_op_sliced& e) const;
      bool operator()(const conditional_op& e) const;
      bool operator()(const binary_op& e) const;
      bool operator()(const unary_op& e) const;
    };

    struct assgn {
      variable lhs_var_;
      std::vector<idx> idxs_;
      expression rhs_;
      assgn();
      assgn(const variable& lhs_var, const std::vector<idx>& idxs,
            const expression& rhs);
      bool lhs_var_occurs_on_rhs() const;
    };

    /**
     * Return the type of the expression indexed by the generalized
     * index sequence.  Return a type with base type
     * <code>ILL_FORMED_T</code> if there are too many indexes.
     *
     * @param[in] e Expression being indexed.
     * @param[in] idxs Index sequence.
     * @return Type of expression applied to indexes.
     */
    expr_type indexed_type(const expression& e,
                           const std::vector<idx>& idxs);

    // FIXME:  is this next dependency necessary?
    // from generator.hpp
    void generate_expression(const expression& e, std::ostream& o);
    void generate_expression(const expression& e, bool user_facing,
                             std::ostream& o);

    bool has_rng_suffix(const std::string& s);
    bool has_lp_suffix(const std::string& s);
    bool is_user_defined(const std::string& name,
                         const std::vector<expression>& args);
    bool is_user_defined_prob_function(const std::string& name,
                                       const expression& variate,
                                       const std::vector<expression>& params);
    bool is_user_defined(const fun& fx);

    struct contains_var : public boost::static_visitor<bool> {
      const variable_map& var_map_;
      explicit contains_var(const variable_map& var_map);
      bool operator()(const nil& e) const;
      bool operator()(const int_literal& e) const;
      bool operator()(const double_literal& e) const;
      bool operator()(const array_expr& e) const;
      bool operator()(const variable& e) const;
      bool operator()(const integrate_ode& e) const;
      bool operator()(const integrate_ode_control& e) const;
      bool operator()(const fun& e) const;
      bool operator()(const index_op& e) const;
      bool operator()(const index_op_sliced& e) const;
      bool operator()(const conditional_op& e) const;
      bool operator()(const binary_op& e) const;
      bool operator()(const unary_op& e) const;
    };

    /**
     * Returns true if the specified expression contains a variable
     * that is defined as a parameter, defined as a transformed
     * parameter, or is a local variable that is not an integer.
     *
     * <p>Compare to <code>has_nonparam_var</code>, which is similar,
     * but excludes variables declared as parameters.
     *
     * @param e Expression to test.
     * @param var_map Variable mapping for origin and types of
     * variables.
     * @return true if expression contains a variable defined as as a
     * parameter, defined as a transformedparameter, or is a local
     * variable that is not an integer.
     */
    bool has_var(const expression& e,
                 const variable_map& var_map);


    struct contains_nonparam_var : public boost::static_visitor<bool> {
      const variable_map& var_map_;
      explicit contains_nonparam_var(const variable_map& var_map);
      bool operator()(const nil& e) const;
      bool operator()(const int_literal& e) const;
      bool operator()(const double_literal& e) const;
      bool operator()(const array_expr& e) const;
      bool operator()(const variable& e) const;
      bool operator()(const integrate_ode& e) const;
      bool operator()(const integrate_ode_control& e) const;
      bool operator()(const fun& e) const;
      bool operator()(const index_op& e) const;
      bool operator()(const index_op_sliced& e) const;
      bool operator()(const conditional_op& e) const;
      bool operator()(const binary_op& e) const;
      bool operator()(const unary_op& e) const;
    };

    /**
     * Returns true if the specified expression contains a variable
     * that is defined as a transformed parameter, or is a local
     * variable that is not an integer.
     *
     * <p>Compare to <code>has_var</code>, which is similar, but
     * includes variables declared as parameters.
     *
     * @param e Expression to test.
     * @param var_map Variable mapping for origin and types of
     * variables.
     * @return true if expression contains a variable defined as a
     * transformed parameter, or is a local variable that is not
     * an integer.
     */
    bool has_non_param_var(const expression& e,
                           const variable_map& var_map);

    bool is_assignable(const expr_type& l_type,
                       const expr_type& r_type,
                       const std::string& failure_message,
                       std::ostream& error_msgs);


    bool ends_with(const std::string& suffix,
                   const std::string& s);


    std::string get_cdf(const std::string& dist_name);

    std::string get_ccdf(const std::string& dist_name);

    std::string get_prob_fun(const std::string& dist_name);

    bool has_prob_fun_suffix(const std::string& name);
    std::string strip_prob_fun_suffix(const std::string& dist_fun);

    bool has_cdf_suffix(const std::string& name);
    std::string strip_cdf_suffix(const std::string& dist_fun);

    bool has_ccdf_suffix(const std::string& name);
    std::string strip_ccdf_suffix(const std::string& dist_fun);

    bool fun_name_exists(const std::string& name);

  }
}
#endif
