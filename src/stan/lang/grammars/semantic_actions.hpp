#ifndef STAN_LANG_GRAMMARS_SEMANTIC_ACTIONS_HPP
#define STAN_LANG_GRAMMARS_SEMANTIC_ACTIONS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/iterator_typedefs.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <boost/spirit/include/qi.hpp>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace stan {

  namespace lang {

    bool has_prob_suffix(const std::string& s);

    void replace_suffix(const std::string& old_suffix,
                        const std::string& new_suffix, fun& f);

    void set_fun_type(fun& fun, std::ostream& error_msgs);

    int num_dimss(std::vector<std::vector<stan::lang::expression> >& dimss);

    /**
     * This is the base class for unnary functors that are adapted to
     * lazy semantic actions by boost::phoenix.  The base class deals
     * with the type dispatch required by Phoenix.
     */
    struct phoenix_functor_unary {
      /**
       * Declare result to be a template struct.
       */
      template <class> struct result;

      /**
       * Specialize as required by Phoenix to functional form
       * with typedef of return type.
       */
      template <typename F, typename T1>
      struct result<F(T1)> { typedef void type; };
    };

    /**
     * This is the base class for binary functors that are adapted to
     * lazy semantic actions by boost::phoenix.  The base class deals
     * with the type dispatch required by Phoenix.
     */
    struct phoenix_functor_binary {
      /**
       * Declare result to be a template struct.
       */
      template <class> struct result;

      /**
       * Specialize as required by Phoenix to functional form
       * with typedef of return type.
       */
      template <typename F, typename T1, typename T2>
      struct result<F(T1, T2)> { typedef void type; };
    };

    /**
     * This is the base class for ternary functors that are adapted to
     * lazy semantic actions by boost::phoenix.  The base class deals
     * with the type dispatch required by Phoenix.
     */
    struct phoenix_functor_ternary {
      /**
       * Declare result to be a template struct.
       */
      template <class> struct result;

      /**
       * Specialize as required by Phoenix to functional form
       * with typedef of return type.
       */
      template <typename F, typename T1, typename T2, typename T3>
      struct result<F(T1, T2, T3)> { typedef void type; };
    };

    /**
     * This is the base class for quatenary functors that are adapted
     * to lazy semantic actions by boost::phoenix.  The base class
     * deals with the type dispatch required by Phoenix.
     */
    struct phoenix_functor_quaternary {
      /**
       * Declare result to be a template struct.
       */
      template <class> struct result;

      /**
       * Specialize as required by Phoenix to functional form
       * with typedef of return type.
       */
      template <typename F, typename T1, typename T2, typename T3, typename T4>
      struct result<F(T1, T2, T3, T4)> { typedef void type; };
    };

    /**
     * This is the base class for quinary functors that are adapted to
     * lazy semantic actions by boost::phoenix.  The base class deals
     * with the type dispatch required by Phoenix.
     */
    struct phoenix_functor_quinary {
      /**
       * Declare result to be a template struct.
       */
      template <class> struct result;

      /**
       * Specialize as required by Phoenix to functional form
       * with typedef of return type.
       */
      template <typename F, typename T1, typename T2, typename T3,
                typename T4, typename T5>
      struct result<F(T1, T2, T3, T4, T5)> { typedef void type; };
    };

    /**
     * This is the base class for senary functors that are adapted to
     * lazy semantic actions by boost::phoenix.  The base class deals
     * with the type dispatch required by Phoenix.
     */
    struct phoenix_functor_senary {
      /**
       * Declare result to be a template struct.
       */
      template <class> struct result;

      /**
       * Specialize as required by Phoenix to functional form
       * with typedef of return type.
       */
      template <typename F, typename T1, typename T2, typename T3,
                typename T4, typename T5, typename T6>
      struct result<F(T1, T2, T3, T4, T5, T6)> { typedef void type; };
    };

    /**
     * This is the base class for septenary functors that are adapted to
     * lazy semantic actions by boost::phoenix.  The base class deals
     * with the type dispatch required by Phoenix.
     */
    struct phoenix_functor_septenary {
      /**
       * Declare result to be a template struct.
       */
      template <class> struct result;

      /**
       * Specialize as required by Phoenix to functional form
       * with typedef of return type.
       */
      template <typename F, typename T1, typename T2, typename T3,
                typename T4, typename T5, typename T6, typename T7>
      struct result<F(T1, T2, T3, T4, T5, T6, T7)> { typedef void type; };
    };


    struct assign_lhs : public phoenix_functor_binary {
      template <typename L, typename R>
      void operator()(L& lhs, const R& rhs) const;
    };
    extern boost::phoenix::function<assign_lhs> assign_lhs_f;


    // called from: expression07_grammar
    struct validate_expr_type3 : public phoenix_functor_ternary {
      void operator()(const expression& expr, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_expr_type3> validate_expr_type3_f;

    // called from: term_grammar
    struct is_prob_fun : public phoenix_functor_binary {
      void operator()(const std::string& s,
                      bool& pass) const;
    };
    extern boost::phoenix::function<is_prob_fun> is_prob_fun_f;

    // called from: expression07_grammar
    struct addition_expr3 : public phoenix_functor_ternary {
      void operator()(expression& expr1, const expression& expr2,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<addition_expr3> addition3_f;

    // called from: expression07_grammar
    struct subtraction_expr3 : public phoenix_functor_ternary {
      void operator()(expression& expr1, const expression& expr2,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<subtraction_expr3> subtraction3_f;

    // called from bare_type_grammar
    struct increment_size_t : public phoenix_functor_unary {
      void operator()(size_t& lhs) const;
    };
    extern boost::phoenix::function<increment_size_t> increment_size_t_f;


    // called from: expression_grammar
    struct validate_conditional_op : public phoenix_functor_quinary {
      void operator()(conditional_op& cond_expr,
                      const scope& var_scope,
                      bool& pass,
                      const variable_map& var_map,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_conditional_op>
    validate_conditional_op_f;


    // called from: expression_grammar
    struct binary_op_expr : public phoenix_functor_quinary {
     void operator()(expression& expr1, const expression& expr2,
                     const std::string& op, const std::string& fun_name,
                     std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<binary_op_expr> binary_op_f;

    // called from: functions_grammar
    struct validate_non_void_arg_function : public phoenix_functor_ternary {
      void operator()(const expr_type& arg_type,
                      bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_non_void_arg_function>
    validate_non_void_arg_f;

    // called from: functions_grammar
    struct set_void_function : public phoenix_functor_quaternary {
      void operator()(const expr_type& return_type, scope& var_scope,
                      bool& pass, std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<set_void_function> set_void_function_f;

    // called from: functions_grammar
    struct set_allows_sampling_origin : public phoenix_functor_binary {
      void operator()(const std::string& identifier, scope& var_scope) const;
    };
    extern boost::phoenix::function<set_allows_sampling_origin>
    set_allows_sampling_origin_f;

    // called from: functions_grammar
    struct validate_declarations : public phoenix_functor_quinary {
      void operator()(bool& pass,
                      std::set<std::pair<std::string,
                                         function_signature_t> >& declared,
                      std::set<std::pair<std::string,
                                         function_signature_t> >& defined,
                      std::ostream& error_msgs,
                      bool allow_undefined) const;
    };
    extern boost::phoenix::function<validate_declarations>
    validate_declarations_f;

    // called from: functions_grammar
    struct add_function_signature : public phoenix_functor_quinary {
      void operator()(const function_decl_def& decl,
              bool& pass,
              std::set<std::pair<std::string,
                                 function_signature_t> >& functions_declared,
              std::set<std::pair<std::string,
                                 function_signature_t> >& functions_defined,
              std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<add_function_signature>
    add_function_signature_f;

    // called from: functions_grammar
    struct validate_return_type : public phoenix_functor_ternary {
      void operator()(function_decl_def& decl, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_return_type>
    validate_return_type_f;

    // called from: functions_grammar
    struct validate_pmf_pdf_variate : public phoenix_functor_ternary {
      void operator()(function_decl_def& decl, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_pmf_pdf_variate>
    validate_pmf_pdf_variate_f;

    // called from: functions_grammar
    struct validate_prob_fun : public phoenix_functor_ternary {
      void operator()(std::string& fname, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_prob_fun> validate_prob_fun_f;


    // called from: functions_grammar
    struct set_fun_params_scope : public phoenix_functor_binary {
      void operator()(scope& var_scope, variable_map& vm) const;
    };
    extern boost::phoenix::function<set_fun_params_scope>
    set_fun_params_scope_f;

    // called from: functions_grammar
    struct unscope_variables : public phoenix_functor_binary {
      void operator()(function_decl_def& decl,
                      variable_map& vm) const;
    };
    extern boost::phoenix::function<unscope_variables> unscope_variables_f;

    // called from: functions_grammar
    struct add_fun_var : public phoenix_functor_quaternary {
      void operator()(arg_decl& decl, bool& pass, variable_map& vm,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<add_fun_var> add_fun_var_f;

    // called from: indexes_grammar
    struct set_omni_idx : public phoenix_functor_unary {
      void operator()(omni_idx& val) const;
    };
    extern boost::phoenix::function<set_omni_idx> set_omni_idx_f;

    // called from: indexes_grammar, statement_grammar
    struct validate_int_expr_silent : public phoenix_functor_binary {
      void operator()(const expression & e, bool& pass) const;
    };
    extern boost::phoenix::function<validate_int_expr_silent>
    validate_int_expr_silent_f;

    // called from: term_grammar
    struct validate_int_expression_warn : public phoenix_functor_ternary {
      void operator()(const expression & e, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_int_expression_warn>
    validate_int_expression_warn_f;

    // called from: indexes_grammar
    struct validate_ints_expression : public phoenix_functor_ternary {
      void operator()(const expression & e, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_ints_expression>
    validate_ints_expression_f;

    // called from: program_grammar
    struct add_params_var : public phoenix_functor_unary {
      void operator()(variable_map& vm) const;
    };
    extern boost::phoenix::function<add_params_var> add_params_var_f;

    // called from: program_grammar
    struct remove_params_var : public phoenix_functor_unary {
      void operator()(variable_map& vm) const;
    };
    extern boost::phoenix::function<remove_params_var> remove_params_var_f;

    // called from: program_grammar
    struct program_error : public phoenix_functor_quinary {
      void operator()(pos_iterator_t _begin, pos_iterator_t _end,
                      pos_iterator_t _where, variable_map& vm,
                      std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<program_error> program_error_f;

    // called from: statement_2_grammar
    struct add_conditional_condition : public phoenix_functor_quaternary {
      void operator()(conditional_statement& cs, const expression& e,
                      bool& pass, std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<add_conditional_condition>
    add_conditional_condition_f;

    // called from: statement_2_grammar
    struct add_conditional_body : public phoenix_functor_binary {
      void operator()(conditional_statement& cs, const statement& s) const;
    };
    extern boost::phoenix::function<add_conditional_body>
    add_conditional_body_f;

    // called from: statement_grammar
    struct deprecate_old_assignment_op : public phoenix_functor_unary {
      void operator()(std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<deprecate_old_assignment_op>
    deprecate_old_assignment_op_f;

    // called from: statement_grammar
    struct validate_return_allowed : public phoenix_functor_ternary {
      void operator()(scope var_scope, bool& pass, std::ostream& error_msgs)
        const;
    };
    extern boost::phoenix::function<validate_return_allowed>
    validate_return_allowed_f;

    // called from: statement_grammar
    struct validate_void_return_allowed : public phoenix_functor_ternary {
      void operator()(scope var_scope, bool& pass, std::ostream& error_msgs)
        const;
    };
    extern boost::phoenix::function<validate_void_return_allowed>
    validate_void_return_allowed_f;

    // called from: statement_grammar
    struct validate_lhs_var_assgn : public phoenix_functor_senary {
      void operator()(const std::string& name, const scope& var_scope,
                      variable& v, bool& pass, const variable_map& vm,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_lhs_var_assgn>
    validate_lhs_var_assgn_f;

    // called from: statement_grammar
    struct validate_assgn : public phoenix_functor_ternary {
      void operator()(const assgn& a, bool& pass, std::ostream& error_msgs)
        const;
    };
    extern boost::phoenix::function<validate_assgn> validate_assgn_f;

    // called from: statement_grammar
    struct validate_assignment : public phoenix_functor_quinary {
      void operator()(assignment& a, const scope& var_scope,
                      bool& pass, variable_map& vm, std::ostream& error_msgs)
        const;
    };
    extern boost::phoenix::function<validate_assignment> validate_assignment_f;

    // called from: statement_grammar
    struct validate_sample : public phoenix_functor_quaternary {
      void operator()(sample& s, const variable_map& var_map,
                      bool& pass, std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_sample> validate_sample_f;

    // called from: statement_grammar
    struct expression_as_statement : public phoenix_functor_ternary {
      void operator()(bool& pass, const stan::lang::expression& expr,
                      std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<expression_as_statement>
    expression_as_statement_f;

    // called from: statement_grammar
    struct unscope_locals : public phoenix_functor_binary {
      void operator()(const std::vector<var_decl>& var_decls, variable_map& vm)
        const;
    };
    extern boost::phoenix::function<unscope_locals> unscope_locals_f;

    // called from: statement_grammar
    struct add_while_condition : public phoenix_functor_quaternary {
      void operator()(while_statement& ws, const expression& e,
                      bool& pass, std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<add_while_condition> add_while_condition_f;

    // called from: statement_grammar
    struct add_while_body : public phoenix_functor_binary {
      void operator()(while_statement& ws, const statement& s) const;
    };
    extern boost::phoenix::function<add_while_body> add_while_body_f;

    // called from: statement_grammar
    struct add_loop_identifier : public phoenix_functor_senary {
      void operator()(const std::string& name,
                      std::string& name_local,
                      const scope& var_scope,
                      bool& pass, variable_map& vm,
                      std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<add_loop_identifier> add_loop_identifier_f;

    // called from: statement_grammar
    struct remove_loop_identifier : public phoenix_functor_binary {
      void operator()(const std::string& name, variable_map& vm) const;
    };
    extern boost::phoenix::function<remove_loop_identifier>
    remove_loop_identifier_f;

    // called from: statement_grammar
    struct deprecate_increment_log_prob : public phoenix_functor_unary {
      void operator()(std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<deprecate_increment_log_prob>
    deprecate_increment_log_prob_f;

    // called from: statement_grammar
    struct validate_allow_sample : public phoenix_functor_ternary {
      void operator()(const scope& var_scope, bool& pass,
                      std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_allow_sample>
    validate_allow_sample_f;

    // called from: statement_grammar
    struct validate_non_void_expression : public phoenix_functor_ternary {
      void operator()(const expression& e, bool& pass, std::ostream& error_msgs)
        const;
    };
    extern boost::phoenix::function<validate_non_void_expression>
    validate_non_void_expression_f;

    // called from: statement_grammar
    struct add_line_number : public phoenix_functor_ternary {
      void operator()(statement& stmt,
                      const pos_iterator_t& begin,
                      const pos_iterator_t& end) const;
    };
    extern boost::phoenix::function<add_line_number> add_line_number_f;

    // called from: statement_grammar
    struct set_void_return : public phoenix_functor_unary {
      void operator()(return_statement& s) const;
    };
    extern boost::phoenix::function<set_void_return> set_void_return_f;

    // called from: statement_grammar
    struct set_no_op : public phoenix_functor_unary {
      void operator()(no_op_statement& s) const;
    };
    extern boost::phoenix::function<set_no_op> set_no_op_f;

    // called from: term_grammar


    struct deprecated_integrate_ode : phoenix_functor_unary {
      void operator()(std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<deprecated_integrate_ode>
    deprecated_integrate_ode_f;

    // test first arguments for both ode calling patterns
    // (with/without control)
    template <class T>
    void validate_integrate_ode_non_control_args(const T& ode_fun,
                                                 const variable_map& var_map,
                                                 bool& pass,
                                                 std::ostream& error_msgs);

    // called from: term_grammar
    struct validate_integrate_ode : public phoenix_functor_quaternary {
      void operator()(const integrate_ode& ode_fun,
                      const variable_map& var_map, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_integrate_ode>
    validate_integrate_ode_f;

    // called from: term_grammar
    struct validate_integrate_ode_control
      : public phoenix_functor_quaternary {
      void operator()(const integrate_ode_control& ode_fun,
                      const variable_map& var_map, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_integrate_ode_control>
    validate_integrate_ode_control_f;

    // called from: term_grammar
    struct set_fun_type_named : public phoenix_functor_quinary {
      void operator()(expression& fun_result, fun& fun,
                      const scope& var_scope, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<set_fun_type_named> set_fun_type_named_f;

    // called from: term_grammar
    struct infer_array_expr_type : public phoenix_functor_senary {
      void operator()(expression& e,
                      array_expr& array_expr,
                      const scope& var_scope,
                      bool& pass,
                      const variable_map& var_map,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<infer_array_expr_type>
    infer_array_expr_type_f;

    // called from: term_grammar
    struct infer_vec_or_matrix_expr_type : public phoenix_functor_senary {
      void operator()(expression& e,
                      row_vector_expr& vec_expr,
                      const scope& var_scope,
                      bool& pass,
                      const variable_map& var_map,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<infer_vec_or_matrix_expr_type>
    infer_vec_or_matrix_expr_type_f;

    // called from: term_grammar
    struct exponentiation_expr : public phoenix_functor_quinary {
      void operator()(expression& expr1, const expression& expr2,
                      const scope& var_scope, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<exponentiation_expr> exponentiation_f;

    // called from: term_grammar
    struct multiplication_expr : public phoenix_functor_ternary {
      void operator()(expression& expr1, const expression& expr2,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<multiplication_expr> multiplication_f;

    // called from: term_grammar
    struct division_expr : public phoenix_functor_ternary {
      void operator()(expression& expr1, const expression& expr2,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<division_expr> division_f;

    // called from: term_grammar
    struct modulus_expr : public phoenix_functor_quaternary {
      void operator()(expression& expr1, const expression& expr2,
                      bool& pass, std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<modulus_expr> modulus_f;

    // called from: term_grammar
    struct left_division_expr : public phoenix_functor_quaternary {
      void operator()(expression& expr1, bool& pass, const expression& expr2,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<left_division_expr> left_division_f;

    // called from: term_grammar
    struct elt_multiplication_expr : public phoenix_functor_ternary {
      void operator()(expression& expr1, const expression& expr2,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<elt_multiplication_expr>
    elt_multiplication_f;

    // called from: term_grammar
    struct elt_division_expr : public phoenix_functor_ternary {
      void operator()(expression& expr1, const expression& expr2,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<elt_division_expr> elt_division_f;

    // called from: term_grammar
    struct negate_expr : public phoenix_functor_quaternary {
      void operator()(expression& expr_result, const expression& expr,
                      bool& pass, std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<negate_expr> negate_expr_f;

    // called from: term_grammar
    struct logical_negate_expr : public phoenix_functor_ternary {
      void operator()(expression& expr_result, const expression& expr,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<logical_negate_expr> logical_negate_expr_f;

    // called from: term_grammar
    struct transpose_expr : public phoenix_functor_ternary {
      void operator()(expression& expr, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<transpose_expr> transpose_f;

    // called from: term_grammar
    struct add_idxs : public phoenix_functor_quaternary {
      void operator()(expression& e, std::vector<idx>& idxs, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<add_idxs> add_idxs_f;

    // called from: term_grammar
    struct add_expression_dimss : public phoenix_functor_quaternary {
      void operator()(expression& expression,
                      std::vector<std::vector<stan::lang::expression> >& dimss,
                      bool& pass, std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<add_expression_dimss>
    add_expression_dimss_f;

    // called from: term_grammar
    struct set_var_type : public phoenix_functor_quinary {
      void operator()(variable& var_expr, expression& val, variable_map& vm,
                      std::ostream& error_msgs, bool& pass) const;
    };
    extern boost::phoenix::function<set_var_type> set_var_type_f;

    struct require_vbar : public phoenix_functor_binary {
      void operator()(bool& pass, std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<require_vbar> require_vbar_f;

    struct validate_no_constraints_vis : public boost::static_visitor<bool> {
      std::stringstream& error_msgs_;
      explicit validate_no_constraints_vis(std::stringstream& error_msgs);
      bool operator()(const nil& /*x*/) const;
      bool operator()(const int_var_decl& x) const;
      bool operator()(const double_var_decl& x) const;
      bool operator()(const vector_var_decl& x) const;
      bool operator()(const row_vector_var_decl& x) const;
      bool operator()(const matrix_var_decl& x) const;
      bool operator()(const unit_vector_var_decl& /*x*/) const;
      bool operator()(const simplex_var_decl& /*x*/) const;
      bool operator()(const ordered_var_decl& /*x*/) const;
      bool operator()(const positive_ordered_var_decl& /*x*/) const;
      bool operator()(const cholesky_factor_var_decl& /*x*/) const;
      bool operator()(const cholesky_corr_var_decl& /*x*/) const;
      bool operator()(const cov_matrix_var_decl& /*x*/) const;
      bool operator()(const corr_matrix_var_decl& /*x*/) const;
    };

    struct data_only_expression : public boost::static_visitor<bool> {
      std::stringstream& error_msgs_;
      variable_map& var_map_;
      data_only_expression(std::stringstream& error_msgs,
                           variable_map& var_map);
      bool operator()(const nil& /*e*/) const;
      bool operator()(const int_literal& /*x*/) const;
      bool operator()(const double_literal& /*x*/) const;
      bool operator()(const array_expr& x) const;
      bool operator()(const matrix_expr& x) const;
      bool operator()(const row_vector_expr& x) const;
      bool operator()(const variable& x) const;
      bool operator()(const integrate_ode& x) const;
      bool operator()(const integrate_ode_control& x) const;
      bool operator()(const fun& x) const;
      bool operator()(const index_op& x) const;
      bool operator()(const index_op_sliced& x) const;
      bool operator()(const conditional_op& x) const;
      bool operator()(const binary_op& x) const;
      bool operator()(const unary_op& x) const;
    };

    // called from: var_decls_grammar
    struct validate_decl_constraints : public phoenix_functor_quinary {
      void operator()(const bool& allow_constraints,
                      const bool& declaration_ok, const var_decl& var_decl,
                      bool& pass, std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_decl_constraints>
    validate_decl_constraints_f;

    // called from: var_decls_grammar
    struct validate_definition : public phoenix_functor_quaternary {
      void operator()(const scope& var_scope, const var_decl& var_decl,
                      bool& pass, std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_definition>
    validate_definition_f;

    struct validate_identifier : public phoenix_functor_ternary {
      std::set<std::string> reserved_word_set_;
      std::set<std::string> const_fun_name_set_;
      validate_identifier();
      void operator()(const std::string& identifier, bool& pass,
                      std::stringstream& error_msgs) const;
      bool contains(const std::set<std::string>& s,
                    const std::string& x) const;
      bool identifier_exists(const std::string& identifier) const;
      void reserve(const std::string& w);
    };
    extern boost::phoenix::function<validate_identifier> validate_identifier_f;

    // copies single dimension from M to N if only M declared
    struct copy_square_cholesky_dimension_if_necessary
      : public phoenix_functor_unary {
      void operator()(cholesky_factor_var_decl& var_decl) const;
    };
    extern boost::phoenix::function<copy_square_cholesky_dimension_if_necessary>
    copy_square_cholesky_dimension_if_necessary_f;

    struct empty_range : public phoenix_functor_binary {
      void operator()(range& r, std::stringstream& /*error_msgs*/) const;
    };
    extern boost::phoenix::function<empty_range> empty_range_f;

    struct validate_int_expr : public phoenix_functor_ternary {
      void operator()(const expression& expr, bool& pass,
                      std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_int_expr> validate_int_expr_f;

    struct set_int_range_lower : public phoenix_functor_quaternary {
      void operator()(range& range, const expression& expr, bool& pass,
                      std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<set_int_range_lower> set_int_range_lower_f;

    struct set_int_range_upper : public phoenix_functor_quaternary {
      void operator()(range& range, const expression& expr, bool& pass,
                      std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<set_int_range_upper> set_int_range_upper_f;

    struct validate_int_data_expr : public phoenix_functor_quinary {
      void operator()(const expression& expr, const scope& var_scope,
                      bool& pass, variable_map& var_map,
                      std::stringstream& error_msgs)
        const;
    };
    extern boost::phoenix::function<validate_int_data_expr>
    validate_int_data_expr_f;

    struct validate_double_expr : public phoenix_functor_ternary {
      void operator()(const expression& expr, bool& pass,
                      std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_double_expr>
    validate_double_expr_f;

    struct set_double_range_lower : public phoenix_functor_quaternary {
      void operator()(range& range, const expression& expr, bool& pass,
                      std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<set_double_range_lower>
    set_double_range_lower_f;

    struct set_double_range_upper : public phoenix_functor_quaternary {
      void operator()(range& range, const expression& expr, bool& pass,
                      std::stringstream& error_msgs) const;
    };
    extern boost::phoenix::function<set_double_range_upper>
    set_double_range_upper_f;

    struct add_var : public phoenix_functor_senary {
      template <typename T>
      void operator()(var_decl& var_decl_result, const T& var_decl,
                      variable_map& vm, bool& pass, const scope& var_scope,
                      std::ostream& error_msgs)
        const;
    };
    extern boost::phoenix::function<add_var> add_var_f;

    struct validate_in_loop : public phoenix_functor_ternary {
      void operator()(bool in_loop, bool& pass, std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<validate_in_loop> validate_in_loop_f;

    struct non_void_expression : public phoenix_functor_ternary {
      void operator()(const expression& e, bool& pass,
                      std::ostream& error_msgs) const;
    };
    extern boost::phoenix::function<non_void_expression> non_void_expression_f;

    struct set_var_scope : public phoenix_functor_binary {
      void operator()(scope& var_scope, const origin_block& program_block)
        const;
    };
    extern boost::phoenix::function<set_var_scope> set_var_scope_f;

    struct set_var_scope_local : public phoenix_functor_binary {
      void operator()(scope& var_scope, const origin_block& program_block)
        const;
    };
    extern boost::phoenix::function<set_var_scope_local>
    set_var_scope_local_f;

    struct reset_var_scope : public phoenix_functor_binary {
      void operator()(scope& var_scope, const scope& scope_enclosing) const;
    };
    extern boost::phoenix::function<reset_var_scope> reset_var_scope_f;

    // handle trace messages as needed for debugging
    struct trace : public phoenix_functor_unary {
      void operator()(const std::string& msg) const;
    };
    extern boost::phoenix::function<trace> trace_f;



  }
}
#endif
