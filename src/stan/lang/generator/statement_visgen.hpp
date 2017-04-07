#ifndef STAN_LANG_GENERATOR_STATEMENT_VISGEN_HPP
#define STAN_LANG_GENERATOR_STATEMENT_VISGEN_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_indent.hpp>
#include <stan/lang/generator/generate_indexed_expr.hpp>
#include <stan/lang/generator/generate_local_var_decls.hpp>
#include <stan/lang/generator/generate_printable.hpp>
#include <stan/lang/generator/visgen.hpp>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    void generate_expression(const expression& e, std::ostream& o);

    void generate_expression(const expression& e, bool user_facing,
                             bool is_var_context, std::ostream& o);

    void generate_idxs(const std::vector<idx>& idxs, std::ostream& o);

    void generate_statement(const statement& s, int indent, std::ostream& o,
                            bool include_sampling, bool is_var_context,
                            bool is_fun_return);

    void generate_statement(const std::vector<statement>& ss, int indent,
                            std::ostream& o, bool include_sampling,
                            bool is_var_context, bool is_fun_return);

    /**
     * Visitor for generating statements.
     */
    struct statement_visgen : public visgen {
      /**
       * Indentation level.
       */
      size_t indent_;

      /**
       * true if sampling statements ae allowed.
       */
      bool include_sampling_;

      /**
       * true if generating inside variable context.
       */
      bool is_var_context_;

      /**
       * true if generating ina  function return context.
       */
      bool is_fun_return_;

      /**
       * Construct a visitor for generating statements at the
       * specified indent level to the specified stream, with flags
       * indicating whether sampling statements are allowed and
       * whether the generation is in a variable context or in a
       * function return context.
       *
       * @param[in] indent indentation level
       * @param[in] include_sampling true if sampling statements are
       * allowed
       * @param[in] is_var_context true if in variable context
       * @param[in] is_fun_return true if in function return context
       * @param[in,out] o stream for generating
       */
      statement_visgen(size_t indent,  bool include_sampling,
                       bool is_var_context, bool is_fun_return,
                       std::ostream& o)
        : visgen(o), indent_(indent), include_sampling_(include_sampling),
          is_var_context_(is_var_context), is_fun_return_(is_fun_return) {  }

      /**
       * Generate the target log density increments for truncating a
       * given density or mass function.
       *
       * @param[in] x sampling statement
       * @param[in] is_user_defined true if user-defined probability
       * function
       * @param[in] prob_fun name of probability function
       */
      void generate_truncation(const sample& x, bool is_user_defined,
                               const std::string& prob_fun) const {
        std::stringstream sso_lp;
        generate_indent(indent_, o_);
        if (x.truncation_.has_low() && x.truncation_.has_high()) {
          // T[L,U]: -log_diff_exp(Dist_cdf_log(U|params),
          //                       Dist_cdf_log(L|Params))
          sso_lp << "log_diff_exp(";
          sso_lp << get_cdf(x.dist_.family_) << "(";
          generate_expression(x.truncation_.high_.expr_, sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            generate_expression(x.dist_.args_[i], sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << "), " << get_cdf(x.dist_.family_) << "(";
          generate_expression(x.truncation_.low_.expr_, sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            generate_expression(x.dist_.args_[i], sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << "))";

        } else if (!x.truncation_.has_low() && x.truncation_.has_high()) {
          // T[,U];  -Dist_cdf_log(U)
          sso_lp << get_cdf(x.dist_.family_) << "(";
          generate_expression(x.truncation_.high_.expr_, sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            generate_expression(x.dist_.args_[i], sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << ")";

        } else if (x.truncation_.has_low() && !x.truncation_.has_high()) {
          // T[L,]: -Dist_ccdf_log(L)
          sso_lp << get_ccdf(x.dist_.family_) << "(";
          generate_expression(x.truncation_.low_.expr_, sso_lp);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            sso_lp << ", ";
            generate_expression(x.dist_.args_[i], sso_lp);
          }
          if (is_user_defined)
            sso_lp << ", pstream__";
          sso_lp << ")";
        }

        o_ << "else lp_accum__.add(-";

        if (x.is_discrete() && x.truncation_.has_low()) {
          o_ << "log_sum_exp(" << sso_lp.str() << ", ";
          // generate adjustment for lower-bound off by 1 due to log CCDF
          o_ << prob_fun << "(";
          generate_expression(x.truncation_.low_.expr_, o_);
          for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
            o_ << ", ";
            generate_expression(x.dist_.args_[i], o_);
          }
          if (is_user_defined) o_ << ", pstream__";
          o_ << "))";
        } else {
          o_ << sso_lp.str();
        }

        o_ << ");" << std::endl;
      }


      void operator()(const nil& /*x*/) const { }

      void operator()(const assignment& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::assign(";
        generate_indexed_expr<true>(x.var_dims_.name_,
                                    x.var_dims_.dims_,
                                    x.var_type_.base_type_,
                                    x.var_type_.dims_.size(),
                                    false,
                                    o_);
        o_ << ", ";
        generate_expression(x.expr_, false, is_var_context_, o_);
        o_ << ");" << EOL;
      }

      void operator()(const assgn& y) const {
        generate_indent(indent_, o_);
        o_ << "stan::model::assign(";

        expression var_expr(y.lhs_var_);
        generate_expression(var_expr, false, is_var_context_, o_);
        o_ << ", "
           << EOL;

        generate_indent(indent_ + 3, o_);
        generate_idxs(y.idxs_, o_);
        o_ << ", "
           << EOL;

        generate_indent(indent_ + 3, o_);
        if (y.lhs_var_occurs_on_rhs()) {
          o_ << "stan::model::deep_copy(";
          generate_expression(y.rhs_, false, is_var_context_, o_);
          o_ << ")";
        } else {
          generate_expression(y.rhs_, false, is_var_context_, o_);
        }

        o_ << ", "
           << EOL;
        generate_indent(indent_ + 3, o_);
        o_ << '"'
           << "assigning variable "
           << y.lhs_var_.name_
           << '"';
        o_ << ");"
           << EOL;
      }

      void operator()(const expression& x) const {
        generate_indent(indent_, o_);
        generate_expression(x, false, is_var_context_, o_);
        o_ << ";" << EOL;
      }

      void operator()(const sample& x) const {
        if (!include_sampling_) return;
        std::string prob_fun = get_prob_fun(x.dist_.family_);
        generate_indent(indent_, o_);
        o_ << "lp_accum__.add(" << prob_fun << "<propto__>(";
        generate_expression(x.expr_, o_);
        for (size_t i = 0; i < x.dist_.args_.size(); ++i) {
          o_ << ", ";
          generate_expression(x.dist_.args_[i], o_);
        }
        bool is_user_defined
          = is_user_defined_prob_function(prob_fun, x.expr_, x.dist_.args_);
        if (is_user_defined)
          o_ << ", pstream__";
        o_ << "));" << EOL;
        // rest of impl is for truncation
        // test variable is within truncation interval
        if (x.truncation_.has_low()) {
          generate_indent(indent_, o_);
          o_ << "if (";
          generate_expression(x.expr_,  o_);
          o_ << " < ";
          generate_expression(x.truncation_.low_.expr_, o_);
          o_ << ") lp_accum__.add(-std::numeric_limits<double>::infinity());"
             << EOL;
        }
        if (x.truncation_.has_high()) {
          generate_indent(indent_, o_);
          if (x.truncation_.has_low()) o_ << "else ";
          o_ << "if (";
          generate_expression(x.expr_, o_);
          o_ << " > ";
          generate_expression(x.truncation_.high_.expr_, o_);
          o_ << ") lp_accum__.add(-std::numeric_limits<double>::infinity());"
             << EOL;
        }
        // generate log denominator for case where bounds test pass
        if (x.truncation_.has_low() || x.truncation_.has_high())
          generate_truncation(x, is_user_defined, prob_fun);
      }

      void operator()(const increment_log_prob_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "lp_accum__.add(";
        generate_expression(x.log_prob_, o_);
        o_ << ");" << EOL;
      }

      void operator()(const statements& x) const {
        bool has_local_vars = x.local_decl_.size() > 0;
        size_t indent = has_local_vars ? (indent_ + 1) : indent_;
        if (has_local_vars) {
          generate_indent(indent_, o_);
          o_ << "{" << EOL;
          generate_local_var_decls(x.local_decl_, indent, o_,
                                   is_var_context_, is_fun_return_);
        }
        o_ << EOL;
        for (size_t i = 0; i < x.statements_.size(); ++i)
          generate_statement(x.statements_[i], indent, o_, include_sampling_,
                             is_var_context_, is_fun_return_);
        if (has_local_vars) {
          generate_indent(indent_, o_);
          o_ << "}" << EOL;
        }
      }

      void operator()(const print_statement& ps) const {
        generate_indent(indent_, o_);
        o_ << "if (pstream__) {" << EOL;
        for (size_t i = 0; i < ps.printables_.size(); ++i) {
          generate_indent(indent_ + 1, o_);
          o_ << "stan_print(pstream__,";
          generate_printable(ps.printables_[i], o_);
          o_ << ");" << EOL;
        }
        generate_indent(indent_ + 1, o_);
        o_ << "*pstream__ << std::endl;" << EOL;
        generate_indent(indent_, o_);
        o_ << '}' << EOL;
      }

      void operator()(const reject_statement& ps) const {
        generate_indent(indent_, o_);
        o_ << "std::stringstream errmsg_stream__;" << EOL;
        for (size_t i = 0; i < ps.printables_.size(); ++i) {
          generate_indent(indent_, o_);
          o_ << "errmsg_stream__ << ";
          generate_printable(ps.printables_[i], o_);
          o_ << ";" << EOL;
        }
        generate_indent(indent_, o_);
        o_ << "throw std::domain_error(errmsg_stream__.str());" << EOL;
      }

      void operator()(const return_statement& rs) const {
        generate_indent(indent_, o_);
        o_ << "return ";
        if (!rs.return_value_.expression_type().is_ill_formed()
            && !rs.return_value_.expression_type().is_void()) {
          o_ << "stan::math::promote_scalar<fun_return_scalar_t__>(";
          generate_expression(rs.return_value_, o_);
          o_ << ")";
        }
        o_ << ";" << EOL;
      }

      void operator()(const for_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "for (int " << x.variable_ << " = ";
        generate_expression(x.range_.low_, o_);
        o_ << "; " << x.variable_ << " <= ";
        generate_expression(x.range_.high_, o_);
        o_ << "; ++" << x.variable_ << ") {" << EOL;
        generate_statement(x.statement_, indent_ + 1, o_, include_sampling_,
                           is_var_context_, is_fun_return_);
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const while_statement& x) const {
        generate_indent(indent_, o_);
        o_ << "while (as_bool(";
        generate_expression(x.condition_, o_);
        o_ << ")) {" << EOL;
        generate_statement(x.body_, indent_+1, o_, include_sampling_,
                           is_var_context_, is_fun_return_);
        generate_indent(indent_, o_);
        o_ << "}" << EOL;
      }

      void operator()(const break_continue_statement& st) const {
        generate_indent(indent_, o_);
        o_ << st.generate_ << ";" << EOL;
      }

      void operator()(const conditional_statement& x) const {
        for (size_t i = 0; i < x.conditions_.size(); ++i) {
          if (i == 0)
            generate_indent(indent_, o_);
          else
            o_ << " else ";
          o_ << "if (as_bool(";
          generate_expression(x.conditions_[i], o_);
          o_ << ")) {" << EOL;
          generate_statement(x.bodies_[i], indent_ + 1, o_, include_sampling_,
                             is_var_context_, is_fun_return_);
          generate_indent(indent_, o_);
          o_ << '}';
        }
        if (x.bodies_.size() > x.conditions_.size()) {
          o_ << " else {" << EOL;
          generate_statement(x.bodies_[x.bodies_.size()-1], indent_ + 1,
                             o_, include_sampling_,
                             is_var_context_, is_fun_return_);
          generate_indent(indent_, o_);
          o_ << '}';
        }
        o_ << EOL;
      }

      void operator()(const no_op_statement& /*x*/) const { }
    };

  }
}
#endif
