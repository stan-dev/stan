#ifndef STAN_LANG_AST_HPP
#define STAN_LANG_AST_HPP

#include <stan/lang/ast/base_expr_type.hpp>
#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/nil.hpp>
#include <stan/lang/ast/printable.hpp>
#include <stan/lang/ast/var_origin.hpp>

#include <stan/lang/ast/fun/infer_type_indexing.hpp>
#include <stan/lang/ast/fun/is_data_origin.hpp>
#include <stan/lang/ast/fun/is_fun_origin.hpp>
#include <stan/lang/ast/fun/print_var_origin.hpp>
#include <stan/lang/ast/fun/total_dims.hpp>
#include <stan/lang/ast/fun/write_base_expr_type.hpp>
#include <stan/lang/ast/fun/is_multi_index.hpp>
#include <stan/lang/ast/fun/is_multi_index_vis.hpp>
#include <stan/lang/ast/fun/is_nil.hpp>
#include <stan/lang/ast/fun/is_nil_vis.hpp>
#include <stan/lang/ast/fun/operator_stream_expr_type.hpp>
#include <stan/lang/ast/fun/promote_primitive.hpp>

#include <stan/lang/ast/sigs/function_signature_t.hpp>
#include <stan/lang/ast/sigs/function_signatures.hpp>

#include <stan/lang/ast/node/array_expr.hpp>
#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/binary_op.hpp>
#include <stan/lang/ast/node/conditional_op.hpp>
#include <stan/lang/ast/node/distribution.hpp>
#include <stan/lang/ast/node/double_literal.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <stan/lang/ast/node/expression_type_vis.hpp>
#include <stan/lang/ast/node/fun.hpp>
#include <stan/lang/ast/node/idx.hpp>
#include <stan/lang/ast/node/index_op.hpp>
#include <stan/lang/ast/node/index_op_sliced.hpp>
#include <stan/lang/ast/node/integrate_ode.hpp>
#include <stan/lang/ast/node/integrate_ode_control.hpp>
#include <stan/lang/ast/node/int_literal.hpp>
#include <stan/lang/ast/node/lb_idx.hpp>
#include <stan/lang/ast/node/lub_idx.hpp>
#include <stan/lang/ast/node/multi_idx.hpp>
#include <stan/lang/ast/node/omni_idx.hpp>
#include <stan/lang/ast/node/range.hpp>
#include <stan/lang/ast/node/statements.hpp>
#include <stan/lang/ast/node/ub_idx.hpp>
#include <stan/lang/ast/node/unary_op.hpp>
#include <stan/lang/ast/node/uni_idx.hpp>
#include <stan/lang/ast/node/variable.hpp>
#include <stan/lang/ast/node/variable_dims.hpp>


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

    struct variable_map {
      typedef std::pair<base_var_decl, var_origin> range_t;

      bool exists(const std::string& name) const;
      base_var_decl get(const std::string& name) const;
      base_expr_type get_base_type(const std::string& name) const;
      size_t get_num_dims(const std::string& name) const;
      var_origin get_origin(const std::string& name) const;
      void add(const std::string& name,
               const base_var_decl& base_decl,
               const var_origin& vo);
      void remove(const std::string& name);
      std::map<std::string, range_t> map_;
    };

    struct int_var_decl : public base_var_decl {
      range range_;
      int_var_decl();
      int_var_decl(range const& range,
                   std::string const& name,
                   std::vector<expression> const& dims,
                   expression const& def);
    };


    struct double_var_decl : public base_var_decl {
      range range_;
      double_var_decl();
      double_var_decl(range const& range,
                      std::string const& name,
                      std::vector<expression> const& dims,
                      expression const& def);
    };

    struct unit_vector_var_decl : public base_var_decl {
      expression K_;
      unit_vector_var_decl();
      unit_vector_var_decl(expression const& K,
                           std::string const& name,
                           std::vector<expression> const& dims);
    };

    struct simplex_var_decl : public base_var_decl {
      expression K_;
      simplex_var_decl();
      simplex_var_decl(expression const& K,
                       std::string const& name,
                       std::vector<expression> const& dims);
    };

    struct ordered_var_decl : public base_var_decl {
      expression K_;
      ordered_var_decl();
      ordered_var_decl(expression const& K,
                       std::string const& name,
                       std::vector<expression> const& dims);
    };

    struct positive_ordered_var_decl : public base_var_decl {
      expression K_;
      positive_ordered_var_decl();
      positive_ordered_var_decl(expression const& K,
                                std::string const& name,
                                std::vector<expression> const& dims);
    };

    struct vector_var_decl : public base_var_decl {
      range range_;
      expression M_;
      vector_var_decl();
      vector_var_decl(range const& range,
                      expression const& M,
                      std::string const& name,
                      std::vector<expression> const& dims,
                      expression const& def);
    };

    struct row_vector_var_decl : public base_var_decl {
      range range_;
      expression N_;
      row_vector_var_decl();
      row_vector_var_decl(range const& range,
                          expression const& N,
                          std::string const& name,
                          std::vector<expression> const& dims,
                          expression const& def);
    };

    struct matrix_var_decl : public base_var_decl {
      range range_;
      expression M_;
      expression N_;
      matrix_var_decl();
      matrix_var_decl(range const& range,
                      expression const& M,
                      expression const& N,
                      std::string const& name,
                      std::vector<expression> const& dims,
                      expression const& def);
    };

    struct cholesky_factor_var_decl : public base_var_decl {
      expression M_;
      expression N_;
      cholesky_factor_var_decl();
      cholesky_factor_var_decl(expression const& M,
                               expression const& N,
                               std::string const& name,
                               std::vector<expression> const& dims);
    };

    struct cholesky_corr_var_decl : public base_var_decl {
      expression K_;
      cholesky_corr_var_decl();
      cholesky_corr_var_decl(expression const& K,
                             std::string const& name,
                             std::vector<expression> const& dims);
    };

    struct cov_matrix_var_decl : public base_var_decl {
      expression K_;
      cov_matrix_var_decl();
      cov_matrix_var_decl(expression const& K,
                          std::string const& name,
                          std::vector<expression> const& dims);
    };


    struct corr_matrix_var_decl : public base_var_decl {
      expression K_;
      corr_matrix_var_decl();
      corr_matrix_var_decl(expression const& K,
                           std::string const& name,
                           std::vector<expression> const& dims);
    };

    struct name_vis : public boost::static_visitor<std::string> {
      name_vis();
      std::string operator()(const nil& x) const;
      std::string operator()(const int_var_decl& x) const;
      std::string operator()(const double_var_decl& x) const;
      std::string operator()(const vector_var_decl& x) const;
      std::string operator()(const row_vector_var_decl& x) const;
      std::string operator()(const matrix_var_decl& x) const;
      std::string operator()(const simplex_var_decl& x) const;
      std::string operator()(const unit_vector_var_decl& x) const;
      std::string operator()(const ordered_var_decl& x) const;
      std::string operator()(const positive_ordered_var_decl& x) const;
      std::string operator()(const cholesky_factor_var_decl& x) const;
      std::string operator()(const cholesky_corr_var_decl& x) const;
      std::string operator()(const cov_matrix_var_decl& x) const;
      std::string operator()(const corr_matrix_var_decl& x) const;
    };

    struct var_decl_base_type_vis
      : public boost::static_visitor<base_var_decl> {
      var_decl_base_type_vis();
      base_var_decl operator()(const nil& x) const;
      base_var_decl operator()(const int_var_decl& x) const;
      base_var_decl operator()(const double_var_decl& x) const;
      base_var_decl operator()(const vector_var_decl& x) const;
      base_var_decl operator()(const row_vector_var_decl& x) const;
      base_var_decl operator()(const matrix_var_decl& x) const;
      base_var_decl operator()(const simplex_var_decl& x) const;
      base_var_decl operator()(const unit_vector_var_decl& x) const;
      base_var_decl operator()(const ordered_var_decl& x) const;
      base_var_decl operator()(const positive_ordered_var_decl& x) const;
      base_var_decl operator()(const cholesky_factor_var_decl& x) const;
      base_var_decl operator()(const cholesky_corr_var_decl& x) const;
      base_var_decl operator()(const cov_matrix_var_decl& x) const;
      base_var_decl operator()(const corr_matrix_var_decl& x) const;
    };

    struct var_decl_dims_vis
      : public boost::static_visitor<std::vector<expression> > {
      var_decl_dims_vis();
      std::vector<expression> operator()(const nil& x) const;
      std::vector<expression> operator()(const int_var_decl& x) const;
      std::vector<expression> operator()(const double_var_decl& x) const;
      std::vector<expression> operator()(const vector_var_decl& x) const;
      std::vector<expression> operator()(const row_vector_var_decl& x) const;
      std::vector<expression> operator()(const matrix_var_decl& x) const;
      std::vector<expression> operator()(const simplex_var_decl& x) const;
      std::vector<expression> operator()(const unit_vector_var_decl& x) const;
      std::vector<expression> operator()(const ordered_var_decl& x) const;
      std::vector<expression> operator()(
                                  const positive_ordered_var_decl& x) const;
      std::vector<expression> operator()(
                                  const cholesky_factor_var_decl& x) const;
      std::vector<expression> operator()(const cholesky_corr_var_decl& x) const;
      std::vector<expression> operator()(const cov_matrix_var_decl& x) const;
      std::vector<expression> operator()(const corr_matrix_var_decl& x) const;
    };

    struct var_decl_has_def_vis
      : public boost::static_visitor<bool> {
      var_decl_has_def_vis();
      bool operator()(const nil& x) const;
      bool operator()(const int_var_decl& x) const;
      bool operator()(const double_var_decl& x) const;
      bool operator()(const vector_var_decl& x) const;
      bool operator()(const row_vector_var_decl& x) const;
      bool operator()(const matrix_var_decl& x) const;
      bool operator()(const simplex_var_decl& x) const;
      bool operator()(const unit_vector_var_decl& x) const;
      bool operator()(const ordered_var_decl& x) const;
      bool operator()(const positive_ordered_var_decl& x) const;
      bool operator()(const cholesky_factor_var_decl& x) const;
      bool operator()(const cholesky_corr_var_decl& x) const;
      bool operator()(const cov_matrix_var_decl& x) const;
      bool operator()(const corr_matrix_var_decl& x) const;
    };

    struct var_decl_def_vis
      : public boost::static_visitor<expression> {
      var_decl_def_vis();
      expression operator()(const nil& x) const;
      expression operator()(const int_var_decl& x) const;
      expression operator()(const double_var_decl& x) const;
      expression operator()(const vector_var_decl& x) const;
      expression operator()(const row_vector_var_decl& x) const;
      expression operator()(const matrix_var_decl& x) const;
      expression operator()(const simplex_var_decl& x) const;
      expression operator()(const unit_vector_var_decl& x) const;
      expression operator()(const ordered_var_decl& x) const;
      expression operator()(const positive_ordered_var_decl& x) const;
      expression operator()(const cholesky_factor_var_decl& x) const;
      expression operator()(const cholesky_corr_var_decl& x) const;
      expression operator()(const cov_matrix_var_decl& x) const;
      expression operator()(const corr_matrix_var_decl& x) const;
    };

    struct var_decl {
      typedef boost::variant<boost::recursive_wrapper<nil>,
                         boost::recursive_wrapper<int_var_decl>,
                         boost::recursive_wrapper<double_var_decl>,
                         boost::recursive_wrapper<vector_var_decl>,
                         boost::recursive_wrapper<row_vector_var_decl>,
                         boost::recursive_wrapper<matrix_var_decl>,
                         boost::recursive_wrapper<simplex_var_decl>,
                         boost::recursive_wrapper<unit_vector_var_decl>,
                         boost::recursive_wrapper<ordered_var_decl>,
                         boost::recursive_wrapper<positive_ordered_var_decl>,
                         boost::recursive_wrapper<cholesky_factor_var_decl>,
                         boost::recursive_wrapper<cholesky_corr_var_decl>,
                         boost::recursive_wrapper<cov_matrix_var_decl>,
                         boost::recursive_wrapper<corr_matrix_var_decl> >
      var_decl_t;
      var_decl_t decl_;
      var_decl();

      // template <typename Decl>
      // var_decl(Decl const& decl);
      var_decl(const var_decl_t& decl);  // NOLINT(runtime/explicit)
      var_decl(const nil& decl);  // NOLINT(runtime/explicit)
      var_decl(const int_var_decl& decl);  // NOLINT(runtime/explicit)
      var_decl(const double_var_decl& decl);  // NOLINT(runtime/explicit)
      var_decl(const vector_var_decl& decl);  // NOLINT(runtime/explicit)
      var_decl(const row_vector_var_decl& decl);  // NOLINT(runtime/explicit)
      var_decl(const matrix_var_decl& decl);  // NOLINT(runtime/explicit)
      var_decl(const simplex_var_decl& decl);  // NOLINT(runtime/explicit)
      var_decl(const unit_vector_var_decl& decl);  // NOLINT(runtime/explicit)
      var_decl(const ordered_var_decl& decl);  // NOLINT(runtime/explicit)
      var_decl(const positive_ordered_var_decl& decl);  // NOLINT
      var_decl(const cholesky_factor_var_decl& decl);  // NOLINT
      var_decl(const cholesky_corr_var_decl& decl);  // NOLINT(runtime/explicit)
      var_decl(const cov_matrix_var_decl& decl);  // NOLINT(runtime/explicit)
      var_decl(const corr_matrix_var_decl& decl);  // NOLINT(runtime/explicit)

      std::string name() const;
      base_var_decl base_decl() const;
      std::vector<expression> dims() const;
      bool has_def() const;
      expression def() const;
    };

    struct statement {
      typedef boost::variant<boost::recursive_wrapper<nil>,
                     boost::recursive_wrapper<assignment>,
                     boost::recursive_wrapper<assgn>,
                     boost::recursive_wrapper<sample>,
                     boost::recursive_wrapper<increment_log_prob_statement>,
                     boost::recursive_wrapper<expression>,
                     boost::recursive_wrapper<statements>,
                     boost::recursive_wrapper<for_statement>,
                     boost::recursive_wrapper<conditional_statement>,
                     boost::recursive_wrapper<while_statement>,
                     boost::recursive_wrapper<break_continue_statement>,
                     boost::recursive_wrapper<print_statement>,
                     boost::recursive_wrapper<reject_statement>,
                     boost::recursive_wrapper<return_statement>,
                     boost::recursive_wrapper<no_op_statement> >
      statement_t;
      statement_t statement_;
      size_t begin_line_;
      size_t end_line_;

      statement();
      statement(const statement_t& st);  // NOLINT(runtime/explicit)
      statement(const nil& st);  // NOLINT(runtime/explicit)
      statement(const assignment& st);  // NOLINT(runtime/explicit)
      statement(const assgn& st);  // NOLINT(runtime/explicit)
      statement(const sample& st);  // NOLINT(runtime/explicit)
      statement(const increment_log_prob_statement& st);  // NOLINT
      statement(const expression& st);  // NOLINT(runtime/explicit)
      statement(const statements& st);  // NOLINT(runtime/explicit)
      statement(const for_statement& st);  // NOLINT(runtime/explicit)
      statement(const conditional_statement& st);  // NOLINT(runtime/explicit)
      statement(const while_statement& st);  // NOLINT(runtime/explicit)
      statement(const break_continue_statement& st);  // NOLINT
      statement(const print_statement& st);  // NOLINT(runtime/explicit)
      statement(const reject_statement& st);  // NOLINT(runtime/explicit)
      statement(const no_op_statement& st);  // NOLINT(runtime/explicit)
      statement(const return_statement& st);  // NOLINT(runtime/explicit)

      bool is_no_op_statement() const;
    };

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
