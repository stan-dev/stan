#ifndef STAN_LANG_GENERATOR_HPP
#define STAN_LANG_GENERATOR_HPP

#include <boost/variant/apply_visitor.hpp>
#include <boost/lexical_cast.hpp>

#include <stan/version.hpp>
#include <stan/lang/ast.hpp>

#include <cstddef>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace stan {

  namespace lang {

    void generate_expression(const expression& e,
                             std::ostream& o);
    void generate_expression(const expression& e,
                             bool user_facing,
                             std::ostream& o);
    void generate_expression(const expression& e,
                             bool user_facing,
                             bool is_var_context,
                             std::ostream& o);
    void generate_bare_type(const expr_type& t,
                            const std::string& scalar_t_name,
                            std::ostream& out);
    void generate_statement(const statement& s,
                            int indent,
                            std::ostream& o,
                            bool include_sampling,
                            bool is_var_context,
                            bool is_fun_return);
    void generate_statement(const std::vector<statement>& ss,
                            int indent,
                            std::ostream& o,
                            bool include_sampling,
                            bool is_var_context,
                            bool is_fun_return);

    const std::string EOL("\n");
    const std::string EOL2("\n\n");
    const std::string INDENT("    ");
    const std::string INDENT2("        ");
    const std::string INDENT3("            ");

    template <typename D>
    bool has_lub(const D& x) {
      return !is_nil(x.range_.low_.expr_) && !is_nil(x.range_.high_.expr_);
    }
    template <typename D>
    bool has_ub(const D& x) {
      return is_nil(x.range_.low_.expr_) && !is_nil(x.range_.high_.expr_);
    }
    template <typename D>
    bool has_lb(const D& x) {
      return !is_nil(x.range_.low_.expr_) && is_nil(x.range_.high_.expr_);
    }

    template <typename T>
    std::string to_string(T i) {
      std::stringstream ss;
      ss << i;
      return ss.str();
    }

    void generate_indent(size_t indent, std::ostream& o) {
      for (size_t k = 0; k < indent; ++k)
        o << INDENT;
    }

    void generate_void_statement(const std::string& name,
                                 const size_t indent,
                                 std::ostream& o)  {
      generate_indent(indent, o);
      o << "(void) " << name << ";  // dummy to suppress unused var warning";
      o << EOL;
    }

    /** generic visitor with output for extension */
    struct visgen {
      typedef void result_type;
      std::ostream& o_;
      explicit visgen(std::ostream& o) : o_(o) { }
    };

    void generate_start_namespace(std::string name,
                                  std::ostream& o) {
      o << "namespace " << name << "_namespace {" << EOL2;
    }

    void generate_end_namespace(std::ostream& o) {
      o << "} // namespace" << EOL2;
    }

    void generate_comment(const std::string& msg, int indent,
                          std::ostream& o) {
      generate_indent(indent, o);
      o << "// " << msg        << EOL;
    }

    /**
     * Print a the specified string to the specified output stream,
     * wrapping in double quotes (") and replacing all double quotes
     * in the input with apostrophes (').  For example, if the input
     * string is <tt>ab"cde"fg</tt> then the string
     * <tt>"ab'cde'fg"</tt> is streamed to the output stream.
     *
     * @param s String to output
     * @param o Output stream
     */
    void generate_quoted_string(const std::string& s,
                                std::ostream& o) {
      o << '"';
      for (size_t i = 0; i < s.size(); ++i) {
        o << ((s[i] == '"') ? '\'' : s[i]);
      }
      o << '"';
    }

    void generate_indexed_expr_user(const std::string& expr,
                                    const std::vector<expression> indexes,
                                    base_expr_type base_type,
                                    std::ostream& o) {
      static const bool user_facing = true;
      o << expr;
      if (indexes.size() == 0) return;
      o << '[';
      for (size_t i = 0; i < indexes.size(); ++i) {
        if (i > 0) o << ", ";
        generate_expression(indexes[i], user_facing, o);
      }
      o << ']';
    }

    template <bool isLHS>
    void generate_indexed_expr(const std::string& expr,
                               const std::vector<expression> indexes,
                               base_expr_type base_type,  // may have more dims
                               size_t e_num_dims,  // array dims
                               bool user_facing,
                               std::ostream& o) {
      if (user_facing) {
        generate_indexed_expr_user(expr, indexes, base_type, o);
        return;
      }
      size_t ai_size = indexes.size();
      if (ai_size == 0) {
        // no indexes
        o << expr;
        return;
      }
      if (ai_size <= (e_num_dims + 1) || base_type != MATRIX_T) {
        for (size_t n = 0; n < ai_size; ++n)
          o << (isLHS ? "get_base1_lhs(" : "get_base1(");
        o << expr;
        for (size_t n = 0; n < ai_size; ++n) {
          o << ',';
          generate_expression(indexes[n], user_facing, o);
          o << ',';
          generate_quoted_string(expr, o);
          o << ',' << (n+1) << ')';
        }
      } else {
        for (size_t n = 0; n < ai_size - 1; ++n)
          o << (isLHS ? "get_base1_lhs(" : "get_base1(");
        o << expr;
        for (size_t n = 0; n < ai_size - 2; ++n) {
          o << ',';
          generate_expression(indexes[n], user_facing, o);
          o << ',';
          generate_quoted_string(expr, o);
          o << ',' << (n+1) << ')';
        }
        o << ',';
        generate_expression(indexes[ai_size - 2U], user_facing, o);
        o << ',';
        generate_expression(indexes[ai_size - 1U], user_facing, o);
        o << ',';
        generate_quoted_string(expr, o);
        o << ',' << (ai_size-1U) << ')';
      }
    }

    // this generates base type for multi-dim expr
    void generate_type(const std::string& base_type,
                       const std::vector<expression>& /*dims*/,
                       size_t end,
                       std::ostream& o) {
      for (size_t i = 0; i < end; ++i) o << "std::vector<";
      o << base_type;
      for (size_t i = 0; i < end; ++i) {
        if (i > 0) o << ' ';
        o << '>';
      }
    }

    /**
     * Generate correct C++ type for expressions which contain a 
     * Stan <code>real</code> variable according to context in
     * which expression is used and expression contents.
     *
     * @param vo expression origin block
     * @param has_var  does expression contains a variable?
     * @param is_var_context true when in auto-diff context
     * @param o generated typename
     */
    void generate_real_var_type(const var_origin& vo,
                                bool has_var,
                                bool is_var_context,
                                std::ostream& o) {
      if (is_fun_origin(vo)) {
        o << "fun_scalar_t__";
      } else if (is_var_context && has_var) {
        o << "T__";
      } else {
        o << "double";
      }
    }

    /**
     * Generate correct C++ type for array expressions
     * according to context in which expression is used.
     * Generated typename may be embedded in angle brackets,
     * add trailing space accordingly.
     *
     * @param base_type expression base type
     * @param real_var_type context-dependent <code>real</code> type
     * @param is_var_context true when in auto-diff context
     * @param o generated typename
     */
    void generate_array_var_type(const base_expr_type base_type,
                                 const std::string& real_var_type,
                                 bool is_var_context,
                                 std::ostream& o) {
      switch (base_type) {
      case INT_T :
        o << "int";
        break;
      case DOUBLE_T :
        o << real_var_type;
        break;
      case VECTOR_T :
        o << (is_var_context ?
              "Eigen::Matrix<T__,Eigen::Dynamic,1> " :"vector_d");
        break;
      case ROW_VECTOR_T :
        o << (is_var_context ?
              "Eigen::Matrix<T__,1,Eigen::Dynamic> " : "row_vector_d");
        break;
      case MATRIX_T :
        o << (is_var_context ?
              "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> " : "matrix_d");
        break;
      }
    }

    void generate_idxs(const std::vector<idx>& idxs,
                       std::ostream& o);
    void generate_idxs_user(const std::vector<idx>& idxs,
                            std::ostream& o);

    struct expression_visgen : public visgen {
      const bool user_facing_;
      const bool is_var_context_;
      explicit expression_visgen(std::ostream& o, bool user_facing,
                                 bool is_var_context)
        : visgen(o),
          user_facing_(user_facing),
          is_var_context_(is_var_context) {
      }
      void operator()(const nil& /*x*/) const {
        o_ << "nil";
      }
      void operator()(const int_literal& n) const { o_ << n.val_; }
      void operator()(const double_literal& x) const {
        std::string num_str = boost::lexical_cast<std::string>(x.val_);
        o_ << num_str;
        if (num_str.find_first_of("eE.") == std::string::npos)
          o_ << ".0";  // trailing 0 to ensure C++ makes it a double
      }
      void operator()(const array_expr& x) const {
        std::stringstream ssRealType;
        generate_real_var_type(x.var_origin_, x.has_var_, is_var_context_,
                               ssRealType);
        std::stringstream ssArrayType;
        generate_array_var_type(x.type_.base_type_, ssRealType.str(),
                                is_var_context_, ssArrayType);
        o_ << "static_cast<";
        generate_type(ssArrayType.str(),
                      x.args_,
                      x.type_.num_dims_,
                      o_);
        o_ << " >(";
        o_ << "stan::math::array_builder<";
        generate_type(ssArrayType.str(),
                      x.args_,
                      x.type_.num_dims_ - 1,
                      o_);
        o_ << " >()";
        for (size_t i = 0; i < x.args_.size(); ++i) {
          o_ << ".add(";
          generate_expression(x.args_[i], user_facing_, is_var_context_, o_);
          o_ << ")";
        }
        o_ << ".array()";
        o_ << ")";
      }
      void operator()(const variable& v) const { o_ << v.name_; }
      void operator()(int n) const {   // NOLINT
        o_ << static_cast<long>(n);    // NOLINT
      }
      void operator()(double x) const { o_ << x; }
      void operator()(const std::string& x) const { o_ << x; }  // identifiers
      void operator()(const index_op& x) const {
        std::stringstream expr_o;
        generate_expression(x.expr_, expr_o);
        std::string expr_string = expr_o.str();
        std::vector<expression> indexes;
        size_t e_num_dims = x.expr_.expression_type().num_dims_;
        base_expr_type base_type = x.expr_.expression_type().base_type_;
        for (size_t i = 0; i < x.dimss_.size(); ++i)
          for (size_t j = 0; j < x.dimss_[i].size(); ++j)
            indexes.push_back(x.dimss_[i][j]);  // wasteful copy, could use refs
        generate_indexed_expr<false>(expr_string, indexes, base_type,
                                     e_num_dims, user_facing_, o_);
      }
      void operator()(const index_op_sliced& x) const {
        if (x.idxs_.size() == 0) {
          generate_expression(x.expr_, user_facing_, o_);
          return;
        }
        if (user_facing_) {
          generate_expression(x.expr_, user_facing_, o_);
          generate_idxs_user(x.idxs_, o_);
          return;
        }
        o_ << "stan::model::rvalue(";
        generate_expression(x.expr_, o_);
        o_ << ", ";
        generate_idxs(x.idxs_, o_);
        o_ << ", ";
        o_ << '"';
        bool user_facing = true;
        generate_expression(x.expr_, user_facing, o_);
        o_ << '"';
        o_ << ")";
      }
      void operator()(const integrate_ode& fx) const {
        o_ << (fx.integration_function_name_ == "integrate_ode"
               ? "integrate_ode_rk45"
               : fx.integration_function_name_)
           << '('
           << fx.system_function_name_
           << "_functor__(), ";

        generate_expression(fx.y0_, o_);
        o_ << ", ";

        generate_expression(fx.t0_, o_);
        o_ << ", ";

        generate_expression(fx.ts_, o_);
        o_ << ", ";

        generate_expression(fx.theta_, o_);
        o_ << ", ";

        generate_expression(fx.x_, o_);
        o_ << ", ";

        generate_expression(fx.x_int_, o_);
        o_ << ", pstream__)";
      }
      void operator()(const integrate_ode_control& fx) const {
        o_ << fx.integration_function_name_
           << '('
           << fx.system_function_name_
           << "_functor__(), ";

        generate_expression(fx.y0_, o_);
        o_ << ", ";

        generate_expression(fx.t0_, o_);
        o_ << ", ";

        generate_expression(fx.ts_, o_);
        o_ << ", ";

        generate_expression(fx.theta_, o_);
        o_ << ", ";

        generate_expression(fx.x_, o_);
        o_ << ", ";

        generate_expression(fx.x_int_, o_);
        o_ << ", pstream__, ";

        generate_expression(fx.rel_tol_, o_);
        o_ << ", ";

        generate_expression(fx.abs_tol_, o_);
        o_ << ", ";

        generate_expression(fx.max_num_steps_, o_);
        o_ << ")";
      }
      void operator()(const fun& fx) const {
        // first test if short-circuit op (binary && and || applied to
        // primitives; overloads are eager, not short-circuiting)
        if (fx.name_ == "logical_or" || fx.name_ == "logical_and") {
          o_ << "(primitive_value(";
          boost::apply_visitor(*this, fx.args_[0].expr_);
          o_ << ") " << ((fx.name_ == "logical_or") ? "||" : "&&")
             << " primitive_value(";
          boost::apply_visitor(*this, fx.args_[1].expr_);
          o_ << "))";
          return;
        }
        o_ << fx.name_ << '(';
        for (size_t i = 0; i < fx.args_.size(); ++i) {
          if (i > 0) o_ << ',';
          boost::apply_visitor(*this, fx.args_[i].expr_);
        }
        if (fx.args_.size() > 0
            && (has_rng_suffix(fx.name_) || has_lp_suffix(fx.name_)))
          o_ << ", ";
        if (has_rng_suffix(fx.name_))
          o_ << "base_rng__";
        if (has_lp_suffix(fx.name_))
          o_ << "lp__, lp_accum__";
        if (is_user_defined(fx)) {
          if (fx.args_.size() > 0
              || has_rng_suffix(fx.name_)
              || has_lp_suffix(fx.name_))
            o_ << ", ";
          o_ << "pstream__";
        }
        o_ << ')';
      }

      void operator()(const conditional_op& expr) const {
        bool types_prim_match
          = (expr.type_.is_primitive() && expr.type_.base_type_ == INT_T)
          || (!expr.has_var_ && expr.type_.is_primitive()
              && (expr.true_val_.expression_type()
                  == expr.false_val_.expression_type()));

        std::stringstream ss;
        generate_real_var_type(expr.var_origin_, expr.has_var_,
                               is_var_context_, ss);

        o_ << "(";
        boost::apply_visitor(*this, expr.cond_.expr_);
        o_ << " ? ";
        if (types_prim_match) {
          boost::apply_visitor(*this, expr.true_val_.expr_);
        } else {
          o_ << "stan::math::promote_scalar<"
             << ss.str()
             << ">(";
          boost::apply_visitor(*this, expr.true_val_.expr_);
          o_ << ")";
        }
        o_ << " : ";
        if (types_prim_match) {
          boost::apply_visitor(*this, expr.false_val_.expr_);
        } else {
          o_ << "stan::math::promote_scalar<"
             << ss.str()
             << ">(";
          boost::apply_visitor(*this, expr.false_val_.expr_);
          o_ << ")";
        }
        o_ << " )";
      }

      void operator()(const binary_op& expr) const {
        o_ << '(';
        boost::apply_visitor(*this, expr.left.expr_);
        o_ << ' ' << expr.op << ' ';
        boost::apply_visitor(*this, expr.right.expr_);
        o_ << ')';
      }
      void operator()(const unary_op& expr) const {
        o_ << expr.op << '(';
        boost::apply_visitor(*this, expr.subject.expr_);
        o_ << ')';
      }
    };     // close struct expression_visgen

    void generate_expression(const expression& e,
                             bool user_facing,
                             bool is_var_context,
                             std::ostream& o) {
      expression_visgen vis(o, user_facing, is_var_context);
      boost::apply_visitor(vis, e.expr_);
    }


    void generate_expression(const expression& e,
                             bool user_facing,
                             std::ostream& o) {
      static const bool is_var_context = false;  // default value
      expression_visgen vis(o, user_facing, is_var_context);
      boost::apply_visitor(vis, e.expr_);
    }

    void generate_expression(const expression& e, std::ostream& o) {
      static const bool user_facing = false;  // default value
      static const bool is_var_context = false;  // default value
      generate_expression(e, user_facing, is_var_context, o);
    }

    static void print_string_literal(std::ostream& o,
                                     const std::string& s) {
      o << '"';
      for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '"' || s[i] == '\\' || s[i] == '\'' )
          o << '\\';
        o << s[i];
      }
      o << '"';
    }

    static void print_quoted_expression(std::ostream& o,
                                        const expression& e) {
      std::stringstream ss;
      generate_expression(e, ss);
      print_string_literal(o, ss.str());
    }

    struct printable_visgen : public visgen {
      explicit printable_visgen(std::ostream& o) : visgen(o) {  }
      void operator()(const std::string& s) const {
        print_string_literal(o_, s);
      }
      void operator()(const expression& e) const {
        generate_expression(e, o_);
      }
    };

    void generate_printable(const printable& p, std::ostream& o) {
      printable_visgen vis(o);
      boost::apply_visitor(vis, p.printable_);
    }

    void generate_using(const std::string& type, std::ostream& o) {
      o << "using " << type << ";" << EOL;
    }

    void generate_using_namespace(const std::string& ns, std::ostream& o) {
      o << "using namespace " << ns << ";" << EOL;
    }


    void generate_usings(std::ostream& o) {
      generate_using("std::istream", o);
      generate_using("std::string", o);
      generate_using("std::stringstream", o);
      generate_using("std::vector", o);
      generate_using("stan::io::dump", o);
      generate_using("stan::math::lgamma", o);
      generate_using("stan::model::prob_grad", o);
      generate_using_namespace("stan::math", o);
      o << EOL;
    }

    void generate_typedef(const std::string& type,
                          const std::string& abbrev,
                          std::ostream& o) {
      o << "typedef" << " " << type << " " << abbrev << ";" << EOL;
    }


    void generate_typedefs(std::ostream& o) {
      generate_typedef("Eigen::Matrix<double,Eigen::Dynamic,1>", "vector_d", o);
      generate_typedef("Eigen::Matrix<double,1,Eigen::Dynamic>",
                       "row_vector_d", o);
      generate_typedef("Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic>",
                       "matrix_d", o);
      o << EOL;
    }

    void generate_include(const std::string& lib_name, std::ostream& o) {
      o << "#include" << " " << "<" << lib_name << ">" << EOL;
    }

    void generate_includes(std::ostream& o) {
      generate_include("stan/model/model_header.hpp", o);
      o << EOL;
    }

    void generate_version_comment(std::ostream& o) {
      o << "// Code generated by Stan version "
        << stan::MAJOR_VERSION  << "." << stan::MINOR_VERSION << EOL2;
    }

    void generate_class_decl(const std::string& model_name,
                             std::ostream& o) {
      o << "class " << model_name << " : public prob_grad {" << EOL;
    }

    void generate_end_class_decl(std::ostream& o) {
      o << "}; // model" << EOL2;
    }

    // use to disambiguate VectorXd(0) ctor from Scalar* alternative
    void generate_eigen_index_expression(const expression& e,
                                         std::ostream& o) {
      o << "static_cast<Eigen::VectorXd::Index>(";
      generate_expression(e.expr_, o);
      o << ")";
    }

    void generate_initializer(std::ostream& o,
                              const std::string& base_type,
                              const std::vector<expression>& dims,
                              const expression& type_arg1 = expression(),
                              const expression& type_arg2 = expression()) {
      for (size_t i = 0; i < dims.size(); ++i) {
        o << '(';
        generate_expression(dims[i].expr_, o);
        o << ',';
        generate_type(base_type, dims, dims.size()- i - 1, o);
      }

      o << '(';
      if (!is_nil(type_arg1)) {
        generate_eigen_index_expression(type_arg1, o);
        if (!is_nil(type_arg2)) {
          o << ',';
          generate_eigen_index_expression(type_arg2, o);
        }
      } else if (!is_nil(type_arg2.expr_)) {
        generate_eigen_index_expression(type_arg2, o);
      } else {
        o << '0';
      }
      o << ')';

      for (size_t i = 0; i < dims.size(); ++i)
        o << ')';
      o << ';' << EOL;
    }

    // only generates the test
    void generate_validate_context_size(std::ostream& o,
                                        const std::string& stage,
                                        const std::string& var_name,
                                        const std::string& base_type,
                                        const std::vector<expression>& dims,
                                        const expression& type_arg1
                                          = expression(),
                                        const expression& type_arg2
                                          = expression()) {
      o << INDENT2
        << "context__.validate_dims("
        << '"' << stage << '"'
        << ", " << '"' << var_name << '"'
        << ", " << '"' << base_type << '"'
        << ", context__.to_vec(";
      for (size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) o << ",";
        generate_expression(dims[i].expr_, o);
      }
      if (!is_nil(type_arg1)) {
        if (dims.size() > 0) o << ",";
        generate_expression(type_arg1.expr_, o);
        if (!is_nil(type_arg2)) {
          o << ",";
          generate_expression(type_arg2.expr_, o);
        }
      }
      o << "));"
        << EOL;
    }

    // see member_var_decl_visgen cut & paste
    struct generate_init_vars_visgen : public visgen {
      int indent_;
      explicit generate_init_vars_visgen(int indent,
                                std::ostream& o)
        : visgen(o),
          indent_(indent) {
      }
      void operator()(const nil& /*x*/) const { }
      void operator()(const int_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_
           << ", std::numeric_limits<int>::min());"
           << EOL;
      }
      void operator()(const double_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const vector_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const row_vector_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const matrix_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const unit_vector_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const simplex_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const ordered_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const positive_ordered_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const cov_matrix_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
      void operator()(const corr_matrix_var_decl& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::fill(" << x.name_ << ",DUMMY_VAR__);" << EOL;
      }
    };

    void generate_init_vars(const std::vector<var_decl>& vs,
                            int indent,
                            std::ostream& o) {
      generate_init_vars_visgen vis(indent, o);
      o << EOL;
      generate_comment("initialize undefined transformed variables to"
                       " avoid seg fault on val access",
                       indent, o);
      for (size_t i = 0; i < vs.size(); ++i) {
        if (!vs[i].has_def()) {
          boost::apply_visitor(vis, vs[i].decl_);
        }
      }
    }

    struct generate_local_var_init_nan_visgen : public visgen {
      const bool is_var_context_;
      const int indent_;
      generate_local_var_init_nan_visgen(bool is_var_context,
                                         int indent,
                                         std::ostream& o)
        : visgen(o),
          is_var_context_(is_var_context),
          indent_(indent) {
      }
      void operator()(const nil& /*x*/) const {
        // no-op
      }
      void operator()(const int_var_decl& x) const {
        // no-op; ints need no init to prevent crashes and no NaN available
      }
      void operator()(const double_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const vector_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const row_vector_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const matrix_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const unit_vector_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const simplex_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const ordered_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        generate_init(x);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        generate_init(x);
      }
      template <typename T>
      void generate_init(const T& x) const {
        generate_indent(indent_, o_);
        o_ << "stan::math::initialize(" << x.name_ << ", "
           << (is_var_context_
               ? "DUMMY_VAR__"
               : "std::numeric_limits<double>::quiet_NaN()")
           << ");"
           << EOL;
      }
    };

    struct var_size_validating_visgen : public visgen {
      const std::string stage_;
      var_size_validating_visgen(std::ostream& o, const std::string& stage)
        : visgen(o),
          stage_(stage) {
      }
      void operator()(const nil& /*x*/) const { }  // dummy
      void operator()(const int_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "int", x.dims_);
      }
      void operator()(const double_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "double", x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "vector_d",
                                       x.dims_, x.M_);
      }
      void operator()(const row_vector_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "row_vector_d",
                                       x.dims_, x.N_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "vector_d",
                                       x.dims_, x.K_);
      }
      void operator()(const simplex_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "vector_d",
                                       x.dims_, x.K_);
      }
      void operator()(const ordered_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "vector_d",
                                       x.dims_, x.K_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "vector_d",
                                       x.dims_, x.K_);
      }
      void operator()(const matrix_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "matrix_d",
                                       x.dims_, x.M_, x.N_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "matrix_d",
                                       x.dims_, x.M_, x.N_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "matrix_d",
                                       x.dims_, x.K_, x.K_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "matrix_d",
                                       x.dims_, x.K_, x.K_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        generate_validate_context_size(o_, stage_, x.name_, "matrix_d",
                                       x.dims_, x.K_, x.K_);
      }
    };


    void generate_validate_positive(const std::string& var_name,
                                    const expression& expr,
                                    std::ostream& o) {
      o << INDENT2;
      o << "validate_non_negative_index(\"" << var_name << "\", ";
      print_quoted_expression(o, expr);
      o << ", ";
      generate_expression(expr, o);
      o << ");" << EOL;
    }

    void generate_initialization(std::ostream& o,
                                 const std::string& var_name,
                                 const std::string& base_type,
                                 const std::vector<expression>& dims,
                                 const expression& type_arg1 = expression(),
                                 const expression& type_arg2 = expression()) {
      // validate all dims are positive
      for (size_t i = 0; i < dims.size(); ++i)
        generate_validate_positive(var_name, dims[i], o);
      if (!is_nil(type_arg1))
        generate_validate_positive(var_name, type_arg1, o);
      if (!is_nil(type_arg2))
        generate_validate_positive(var_name, type_arg2, o);

      // initialize variable
      o << INDENT2
        << var_name << " = ";
      generate_type(base_type, dims, dims.size(), o);
      generate_initializer(o, base_type, dims, type_arg1, type_arg2);
    }

    struct var_resizing_visgen : public visgen {
      explicit var_resizing_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const { }  // dummy
      void operator()(const int_var_decl& x) const {
        generate_initialization(o_, x.name_, "int", x.dims_,
                                nil(), nil());
      }
      void operator()(const double_var_decl& x) const {
        generate_initialization(o_, x.name_, "double", x.dims_, nil(), nil());
      }
      void operator()(const vector_var_decl& x) const {
        generate_initialization(o_, x.name_, "vector_d", x.dims_, x.M_, nil());
      }
      void operator()(const row_vector_var_decl& x) const {
        generate_initialization(o_, x.name_, "row_vector_d", x.dims_, x.N_,
                                nil());
      }
      void operator()(const unit_vector_var_decl& x) const {
        generate_initialization(o_, x.name_, "vector_d", x.dims_, x.K_, nil());
      }
      void operator()(const simplex_var_decl& x) const {
        generate_initialization(o_, x.name_, "vector_d", x.dims_, x.K_, nil());
      }
      void operator()(const ordered_var_decl& x) const {
        generate_initialization(o_, x.name_, "vector_d", x.dims_, x.K_, nil());
      }
      void operator()(const positive_ordered_var_decl& x) const {
        generate_initialization(o_, x.name_, "vector_d", x.dims_, x.K_, nil());
      }
      void operator()(const matrix_var_decl& x) const {
        generate_initialization(o_, x.name_, "matrix_d",
                                x.dims_, x.M_, x.N_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        generate_initialization(o_, x.name_, "matrix_d", x.dims_, x.M_, x.N_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        generate_initialization(o_, x.name_, "matrix_d", x.dims_, x.K_, x.K_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        generate_initialization(o_, x.name_, "matrix_d", x.dims_, x.K_, x.K_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        generate_initialization(o_, x.name_, "matrix_d", x.dims_, x.K_, x.K_);
      }
    };

    void generate_var_resizing(const std::vector<var_decl>& vs,
                               std::ostream& o) {
      var_resizing_visgen vis_resizer(o);
      generate_init_vars_visgen vis_filler(2, o);
      for (size_t i = 0; i < vs.size(); ++i) {
        boost::apply_visitor(vis_resizer, vs[i].decl_);
        boost::apply_visitor(vis_filler, vs[i].decl_);
        if (vs[i].has_def()) {
          o << INDENT2 << "stan::math::assign("
            << vs[i].name()
            << ",";
          generate_expression(vs[i].def(), o);
          o << ");" << EOL;
        }
      }
    }

    const std::vector<expression> EMPTY_EXP_VECTOR(0);

    struct init_local_var_visgen : public visgen {
      const bool declare_vars_;
      const bool is_var_context_;
      explicit init_local_var_visgen(bool declare_vars,
                            bool is_var_context,
                            std::ostream& o)
        : visgen(o),
          declare_vars_(declare_vars),
          is_var_context_(is_var_context) {
      }
      template <typename D>
      void generate_initialize_array_bounded(const D& x,
                                             const std::string& base_type,
                                             const std::string& read_fun_prefix,
                                const std::vector<expression>& dim_args) const {
        std::vector<expression> read_args;
        std::string read_fun(read_fun_prefix);
        if (has_lub(x)) {
          read_fun += "_lub";
          read_args.push_back(x.range_.low_);
          read_args.push_back(x.range_.high_);
        } else if (has_lb(x)) {
          read_fun += "_lb";
          read_args.push_back(x.range_.low_);
        } else if (has_ub(x)) {
          read_fun += "_ub";
          read_args.push_back(x.range_.high_);
        }
        for (size_t i = 0; i < dim_args.size(); ++i)
          read_args.push_back(dim_args[i]);
        generate_initialize_array(base_type, read_fun, read_args,
                                    x.name_, x.dims_);
      }
      void operator()(const nil& /*x*/) const { }
      void operator()(const int_var_decl& x) const {
        generate_initialize_array("int", "integer", EMPTY_EXP_VECTOR,
                                  x.name_, x.dims_);
      }
      void operator()(const double_var_decl& x) const {
        std::vector<expression> read_args;
        generate_initialize_array_bounded(x,
                                          is_var_context_ ? "T__" : "double",
                                          "scalar", read_args);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        generate_initialize_array_bounded(x,
                                          is_var_context_
                                          ? "Eigen::Matrix"
                                          "<T__,Eigen::Dynamic,1> "
                                          : "vector_d",
                                          "vector", read_args);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.N_);
        generate_initialize_array_bounded(x,
                                          is_var_context_
                                          ? "Eigen::Matrix"
                                          "<T__,1,Eigen::Dynamic> "
                                          : "row_vector_d",
                                          "row_vector", read_args);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        read_args.push_back(x.N_);
        generate_initialize_array_bounded(x,
                                          is_var_context_
                                          ? "Eigen::Matrix"
                                          "<T__,Eigen::Dynamic,Eigen::Dynamic> "
                                          : "matrix_d",
                                          "matrix", read_args);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix"
                                  "<T__,Eigen::Dynamic,1> "
                                  : "vector_d",
                                  "unit_vector", read_args, x.name_, x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix"
                                  "<T__,Eigen::Dynamic,1> "
                                  : "vector_d",
                                  "simplex", read_args, x.name_, x.dims_);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix"
                                  "<T__,Eigen::Dynamic,1> "
                                  : "vector_d",
                                  "ordered", read_args, x.name_, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix"
                                  "<T__,Eigen::Dynamic,1> "
                                  : "vector_d",
                                  "positive_ordered", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        read_args.push_back(x.N_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix"
                                  "<T__,Eigen::Dynamic,Eigen::Dynamic> "
                                  : "matrix_d",
                                  "cholesky_factor", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix"
                                  "<T__,Eigen::Dynamic,Eigen::Dynamic> "
                                  : "matrix_d",
                                  "cholesky_corr", read_args, x.name_, x.dims_);
      }

      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix"
                                  "<T__,Eigen::Dynamic,Eigen::Dynamic> "
                                  : "matrix_d",
                                  "cov_matrix", read_args, x.name_, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array(is_var_context_
                                  ? "Eigen::Matrix"
                                  "<T__,Eigen::Dynamic,Eigen::Dynamic> "
                                  : "matrix_d",
                                  "corr_matrix", read_args, x.name_, x.dims_);
      }


      void generate_initialize_array(const std::string& var_type,
                                     const std::string& read_type,
                                     const std::vector<expression>& read_args,
                                     const std::string& name,
                                     const std::vector<expression>& dims)
      const {
        if (declare_vars_) {
          o_ << INDENT2;
          for (size_t i = 0; i < dims.size(); ++i) o_ << "vector<";
          o_ << var_type;
          for (size_t i = 0; i < dims.size(); ++i) o_ << "> ";
          if (dims.size() == 0) o_ << " ";
          o_ << name << ";" << EOL;
        }

        if (dims.size() == 0) {
          generate_void_statement(name, 2, o_);
          o_ << INDENT2 << "if (jacobian__)" << EOL;

          // w Jacobian
          generate_indent(3, o_);
          o_ << name << " = in__." << read_type  << "_constrain(";
          for (size_t j = 0; j < read_args.size(); ++j) {
            if (j > 0) o_ << ",";
            generate_expression(read_args[j], o_);
          }
          if (read_args.size() > 0)
            o_ << ",";
          o_ << "lp__";
          o_ << ");" << EOL;

          o_ << INDENT2 << "else" << EOL;

          // w/o Jacobian
          generate_indent(3, o_);
          o_ << name << " = in__." << read_type  << "_constrain(";
          for (size_t j = 0; j < read_args.size(); ++j) {
            if (j > 0) o_ << ",";
            generate_expression(read_args[j], o_);
          }
          o_ << ");" << EOL;

        } else {
          // dims > 0
          std::string name_dims(name);
          for (size_t i = 0; i < dims.size(); ++i) {
            generate_indent(i + 2, o_);
            o_ << "size_t dim_"  << name << "_" << i << "__ = ";
            generate_expression(dims[i], o_);
            o_ << ";" << EOL;

            if (i < dims.size() - 1) {
              generate_indent(i + 2, o_);
              o_ << name_dims << ".resize(dim" << "_"
                 << name << "_" << i << "__);"
                 << EOL;
              name_dims.append("[k_").append(to_string(i)).append("__]");
            }

            generate_indent(i + 2, o_);
            if (i == dims.size() - 1) {
              o_ << name_dims << ".reserve(dim_" << name
                 << "_" << i << "__);" << EOL;
              generate_indent(i + 2, o_);
            }

            o_ << "for (size_t k_" << i << "__ = 0;"
               << " k_" << i << "__ < dim_" << name << "_" << i << "__;"
               << " ++k_" << i << "__) {" << EOL;

            // if on the last loop, push read element into array
            if (i == dims.size() - 1) {
              generate_indent(i + 3, o_);
              o_ << "if (jacobian__)" << EOL;

              // w Jacobian
              generate_indent(i + 4, o_);
              o_ << name_dims << ".push_back(in__."
                 << read_type << "_constrain(";
              for (size_t j = 0; j < read_args.size(); ++j) {
                if (j > 0) o_ << ",";
                generate_expression(read_args[j], o_);
              }
              if (read_args.size() > 0)
                o_ << ",";
              o_ << "lp__";
              o_ << "));" << EOL;

              generate_indent(i + 3, o_);
              o_ << "else" << EOL;

              // w/o Jacobian
              generate_indent(i + 4, o_);
              o_ << name_dims << ".push_back(in__."
                 << read_type << "_constrain(";
              for (size_t j = 0; j < read_args.size(); ++j) {
                if (j > 0) o_ << ",";
                generate_expression(read_args[j], o_);
              }
              o_ << "));" << EOL;
            }
          }

          for (size_t i = dims.size(); i > 0; --i) {
            generate_indent(i + 1, o_);
            o_ << "}" << EOL;
          }
        }
        o_ << EOL;
      }
    };


    void generate_local_var_inits(std::vector<var_decl> vs,
                                  bool is_var_context,
                                  bool declare_vars,
                                  std::ostream& o) {
      o << INDENT2
        << "stan::io::reader<"
        << (is_var_context ? "T__" : "double")
        << "> in__(params_r__,params_i__);" << EOL2;
      init_local_var_visgen vis_init(declare_vars, is_var_context, o);
      for (size_t i = 0; i < vs.size(); ++i) {
          boost::apply_visitor(vis_init, vs[i].decl_);
      }
    }


    void generate_public_decl(std::ostream& o) {
      o << "public:" << EOL;
    }

    void generate_private_decl(std::ostream& o) {
      o << "private:" << EOL;
    }

    struct validate_var_decl_visgen : public visgen {
      int indents_;
      validate_var_decl_visgen(int indents,
                               std::ostream& o)
        : visgen(o),
          indents_(indents) {
      }
      void generate_begin_for_dims(const std::vector<expression>& dims)
        const {
        for (size_t i = 0; i < dims.size(); ++i) {
          generate_indent(indents_+i, o_);
          o_ << "for (int k" << i << "__ = 0;"
             << " k" << i << "__ < ";
          generate_expression(dims[i].expr_, o_);
          o_ << ";";
          o_ << " ++k" << i << "__) {" << EOL;
        }
      }
      void generate_end_for_dims(size_t dims_size) const {
        for (size_t i = 0; i < dims_size; ++i) {
          generate_indent(indents_ + dims_size - i - 1, o_);
          o_ << "}" << EOL;
        }
      }
      void generate_loop_var(const std::string& name,
                             size_t dims_size) const {
        o_ << name;
        for (size_t i = 0; i < dims_size; ++i)
          o_ << "[k" << i << "__]";
      }
      void operator()(const nil& /*x*/) const { }
      template <typename T>
      void basic_validate(const T& x) const {
        if (!(x.range_.has_low() || x.range_.has_high())) {
          return;  // unconstrained
        }
        generate_begin_for_dims(x.dims_);
        if (x.range_.has_low()) {
          generate_indent(indents_ + x.dims_.size(), o_);
          o_ << "check_greater_or_equal(function__,";
          o_ << "\"";
          generate_loop_var(x.name_, x.dims_.size());
          o_ << "\",";
          generate_loop_var(x.name_, x.dims_.size());
          o_ << ",";
          generate_expression(x.range_.low_.expr_, o_);
          o_ << ");" << EOL;
        }
        if (x.range_.has_high()) {
          generate_indent(indents_ + x.dims_.size(), o_);
          o_ << "check_less_or_equal(function__,";
          o_ << "\"";
          generate_loop_var(x.name_, x.dims_.size());
          o_ << "\",";
          generate_loop_var(x.name_, x.dims_.size());
          o_ << ",";
          generate_expression(x.range_.high_.expr_, o_);
          o_ << ");" << EOL;
        }
        generate_end_for_dims(x.dims_.size());
      }
      void operator()(const int_var_decl& x) const {
        basic_validate(x);
      }
      void operator()(const double_var_decl& x) const {
        basic_validate(x);
      }
      void operator()(const vector_var_decl& x) const {
        basic_validate(x);
      }
      void operator()(const row_vector_var_decl& x) const {
        basic_validate(x);
      }
      void operator()(const matrix_var_decl& x) const {
        basic_validate(x);
      }
      template <typename T>
      void nonbasic_validate(const T& x,
                             const std::string& type_name) const {
        generate_begin_for_dims(x.dims_);
        generate_indent(indents_ + x.dims_.size(), o_);
        o_ << "stan::math::check_" << type_name << "(function__,";
        o_ << "\"";
        generate_loop_var(x.name_, x.dims_.size());
        o_ << "\",";
        generate_loop_var(x.name_, x.dims_.size());
        o_ << ");"
           << EOL;
        generate_end_for_dims(x.dims_.size());
      }
      void operator()(const unit_vector_var_decl& x) const {
        nonbasic_validate(x, "unit_vector");
      }
      void operator()(const simplex_var_decl& x) const {
        nonbasic_validate(x, "simplex");
      }
      void operator()(const ordered_var_decl& x) const {
        nonbasic_validate(x, "ordered");
      }
      void operator()(const positive_ordered_var_decl& x) const {
        nonbasic_validate(x, "positive_ordered");
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        nonbasic_validate(x, "cholesky_factor");
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        nonbasic_validate(x, "cholesky_factor_corr");
      }
      void operator()(const cov_matrix_var_decl& x) const {
        nonbasic_validate(x, "cov_matrix");
      }
      void operator()(const corr_matrix_var_decl& x) const {
        nonbasic_validate(x, "corr_matrix");
      }
    };


    void generate_validate_var_decl(const var_decl& decl,
                                    int indent,
                                    std::ostream& o) {
      validate_var_decl_visgen vis(indent, o);
      boost::apply_visitor(vis, decl.decl_);
    }

    void generate_validate_var_decls(const std::vector<var_decl> decls,
                                     int indent,
                                     std::ostream& o) {
      for (size_t i = 0; i < decls.size(); ++i)
        generate_validate_var_decl(decls[i], indent, o);
    }

    // see _var_decl_visgen cut & paste
    struct member_var_decl_visgen : public visgen {
      int indents_;
      member_var_decl_visgen(int indents,
                             std::ostream& o)
        : visgen(o),
          indents_(indents) {
      }
      void operator()(const nil& /*x*/) const { }
      void operator()(const int_var_decl& x) const {
        declare_array("int", x.name_, x.dims_.size());
      }
      void operator()(const double_var_decl& x) const {
        declare_array("double", x.name_, x.dims_.size());
      }
      void operator()(const unit_vector_var_decl& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }
      void operator()(const simplex_var_decl& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }
      void operator()(const ordered_var_decl& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }
      void operator()(const positive_ordered_var_decl& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }
      void operator()(const cov_matrix_var_decl& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }
      void operator()(const corr_matrix_var_decl& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }
      void operator()(const vector_var_decl& x) const {
        declare_array(("vector_d"), x.name_, x.dims_.size());
      }
      void operator()(const row_vector_var_decl& x) const {
        declare_array(("row_vector_d"), x.name_, x.dims_.size());
      }
      void operator()(const matrix_var_decl& x) const {
        declare_array(("matrix_d"), x.name_, x.dims_.size());
      }
      void declare_array(const std::string& type,
                         const std::string& name,
                         size_t size) const {
        for (int i = 0; i < indents_; ++i)
          o_ << INDENT;
        for (size_t i = 0; i < size; ++i) {
          o_ << "vector<";
        }
        o_ << type;
        if (size > 0) {
          o_ << ">";
        }
        for (size_t i = 1; i < size; ++i) {
          o_ << " >";
        }
        o_ << " " << name << ";" << EOL;
      }
    };

    void generate_member_var_decls(const std::vector<var_decl>& vs,
                                   int indent,
                                   std::ostream& o) {
      member_var_decl_visgen vis(indent, o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
    }

    // see member_var_decl_visgen cut & paste
    struct local_var_decl_visgen : public visgen {
      int indents_;
      bool is_var_context_;
      bool is_fun_return_;
      explicit local_var_decl_visgen(int indents,
                            bool is_var_context,
                            bool is_fun_return,
                            std::ostream& o)
        : visgen(o),
          indents_(indents),
          is_var_context_(is_var_context),
          is_fun_return_(is_fun_return) {
      }
      void operator()(const nil& /*x*/) const { }
      void operator()(const int_var_decl& x) const {
        std::vector<expression> ctor_args;
        declare_array("int", ctor_args, x.name_, x.dims_);
      }
      void operator()(const double_var_decl& x) const {
        std::vector<expression> ctor_args;
        declare_array(is_fun_return_
                      ? "fun_scalar_t__"
                      : (is_var_context_ ? "T__" : "double"),
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.M_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,Eigen::Dynamic,1> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,1> " : "vector_d"),
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.N_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,1,Eigen::Dynamic> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,1,Eigen::Dynamic> "
                         : "row_vector_d"),
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.M_);
        ctor_args.push_back(x.N_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,"
                      "Eigen::Dynamic,Eigen::Dynamic> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                         : "matrix_d"),
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,Eigen::Dynamic,1> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,1> " : "vector_d"),
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,Eigen::Dynamic,1> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,1> " : "vector_d"),
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,Eigen::Dynamic,1> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,1> " : "vector_d"),
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,Eigen::Dynamic,1> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,1> " : "vector_d"),
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.M_);
        ctor_args.push_back(x.N_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,"
                      "Eigen::Dynamic,Eigen::Dynamic> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                         : "matrix_d"),
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        ctor_args.push_back(x.K_);
        declare_array(is_var_context_
                      ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                      : "matrix_d",
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,"
                      "Eigen::Dynamic,Eigen::Dynamic> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                         : "matrix_d"),
                      ctor_args, x.name_, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> ctor_args;
        ctor_args.push_back(x.K_);
        ctor_args.push_back(x.K_);
        declare_array(is_fun_return_
                      ? "Eigen::Matrix<fun_scalar_t__,"
                      "Eigen::Dynamic,Eigen::Dynamic> "
                      : (is_var_context_
                         ? "Eigen::Matrix<T__,Eigen::Dynamic,Eigen::Dynamic> "
                         : "matrix_d"),
                      ctor_args, x.name_, x.dims_);
      }
      void generate_type(const std::string& type,
                         size_t num_dims) const {
        for (size_t i = 0; i < num_dims; ++i)
          o_ << "vector<";
        o_ << type;
        for (size_t i = 0; i < num_dims; ++i) {
          if (i > 0) o_ << " ";
          o_ << ">";
        }
      }

      void generate_void_statement(const std::string& name) const {
        o_ << "(void) " << name << ";  // dummy to suppress unused var warning";
      }

      // var_decl     -> type[0] name init_args[0] ;
      // init_args[k] -> ctor_args  if no dims left
      // init_args[k] -> ( dim[k] , ( type[k+1] init_args[k+1] ) )
      void generate_init_args(const std::string& type,
                              const std::vector<expression>& ctor_args,
                              const std::vector<expression>& dims,
                              size_t dim) const {
        if (dim < dims.size()) {  // more dims left
          o_ << '(';  // open(1)
          generate_expression(dims[dim], o_);
          if ((dim + 1 < dims.size()) ||  ctor_args.size() > 0) {
            o_ << ", (";  // open(2)
            generate_type(type, dims.size() - dim - 1);
            generate_init_args(type, ctor_args, dims, dim + 1);
            o_ << ')';  // close(2)
          } else if (type == "var") {
            o_ << ", DUMMY_VAR__";
          } else if (type == "int") {
            o_ << ", 0";
          } else if (type == "double") {
            o_ << ", 0.0";
          } else {
            // shouldn't hit this
          }
          o_ << ')';  // close(1)
        } else {
          if (ctor_args.size() == 0) {  // scalar int or real
            if (type == "int") {
              o_ << "(0)";
            } else if (type == "double") {
              o_ << "(0.0)";
            } else if (type == "var") {
              o_ << "(DUMMY_VAR__)";
            } else {
              // shouldn't hit this, either
            }
          } else if (ctor_args.size() == 1) {  // vector
            o_ << '(';
            generate_eigen_index_expression(ctor_args[0], o_);
            o_ << ')';
          } else if (ctor_args.size() > 1) {  // matrix
            o_ << '(';
            generate_eigen_index_expression(ctor_args[0], o_);
            o_ << ',';
            generate_eigen_index_expression(ctor_args[1], o_);
            o_ << ')';
          }
        }
      }
      void declare_array(const std::string& type,
                         const std::vector<expression>& ctor_args,
                         const std::string& name,
                         const std::vector<expression>& dims,
                         const expression& definition = expression()) const {
        // require double parens to counter "most vexing parse" problem
        generate_indent(indents_, o_);
        generate_type(type, dims.size());
        o_ << ' '  << name;
        generate_init_args(type, ctor_args, dims, 0);
        o_ << ";" << EOL;
        if (dims.size() == 0) {
          generate_indent(indents_, o_);
          generate_void_statement(name);
          o_ << EOL;
        }
        if (type == "Eigen::Matrix<T__, Eigen::Dynamic, Eigen::Dynamic> "
            || type == "Eigen::Matrix<T__, 1, Eigen::Dynamic> "
            || type == "Eigen::Matrix<T__, Eigen::Dynamic, 1> ") {
          generate_indent(indents_, o_);
          o_ << "stan::math::fill(" << name << ", DUMMY_VAR__);" << EOL;
        }
      }
    };

    void generate_local_var_decls(const std::vector<var_decl>& vs,
                                  int indent,
                                  std::ostream& o,
                                  bool is_var_context,
                                  bool is_fun_return) {
      local_var_decl_visgen vis_decl(indent, is_var_context, is_fun_return, o);
      generate_local_var_init_nan_visgen vis_init(is_var_context, indent, o);
      generate_init_vars_visgen vis_filler(indent, o);
      for (size_t i = 0; i < vs.size(); ++i) {
        boost::apply_visitor(vis_decl, vs[i].decl_);
        boost::apply_visitor(vis_init, vs[i].decl_);
        boost::apply_visitor(vis_filler, vs[i].decl_);
        if (vs[i].has_def()) {
          generate_indent(indent, o);
          o << "stan::math::assign("
            << vs[i].name()
            << ",";
          generate_expression(vs[i].def(), false, is_var_context, o);
          o << ");" << EOL;
        }
      }
      o << EOL;
    }


    void generate_local_var_init_nan(const std::vector<var_decl>& vs,
                                     int indent,
                                     std::ostream& o,
                                     bool is_var_context,
                                     bool is_fun_return) {
      generate_local_var_init_nan_visgen vis(is_var_context, indent, o);
      for (size_t i = 0; i < vs.size(); ++i) {
        boost::apply_visitor(vis, vs[i].decl_);
      }
    }


    void generate_define_vars(const std::vector<var_decl>& vs,
                              int indent,
                              bool is_var_context,
                              std::ostream& o) {
      generate_comment("assign variable definitions",
                       indent, o);
      for (size_t i = 0; i < vs.size(); ++i) {
        if (vs[i].has_def()) {
          generate_indent(indent, o);
          o << "stan::math::assign("
            << vs[i].name()
            << ",";
          generate_expression(vs[i].def(), false, is_var_context, o);
          o << ");" << EOL;
        }
      }
    }

    struct validate_transformed_params_visgen : public visgen {
      int indents_;
      validate_transformed_params_visgen(int indents,
                                         std::ostream& o)
        : visgen(o),
          indents_(indents)
      { }
      void operator()(const nil& /*x*/) const { }
      void operator()(const int_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        validate_array(x.name_, dims, 0);
      }
      void operator()(const double_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        validate_array(x.name_, dims, 0);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.M_);
        validate_array(x.name_, dims, 1);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 1);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 1);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 1);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 1);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.N_);
        validate_array(x.name_, dims, 1);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.M_);
        dims.push_back(x.N_);
        validate_array(x.name_, dims, 2);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.M_);
        dims.push_back(x.N_);
        validate_array(x.name_, dims, 2);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 2);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 2);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        dims.push_back(x.K_);
        validate_array(x.name_, dims, 2);
      }
      void validate_array(const std::string& name,
                          const std::vector<expression>& dims,
                          size_t matrix_dims) const {
        size_t non_matrix_dims = dims.size() - matrix_dims;

        for (size_t k = 0; k < dims.size(); ++k) {
          generate_indent(indents_ + k, o_);
          o_ << "for (int i" << k << "__ = 0; i" << k << "__ < ";
          generate_expression(dims[k], o_);
          o_ << "; ++i" << k << "__) {" << EOL;
        }

        generate_indent(indents_ + dims.size(), o_);
        o_ << "if (stan::math::is_uninitialized(" << name;
        for (size_t k = 0; k < non_matrix_dims; ++k)
          o_ << "[i" << k << "__]";
        if (matrix_dims > 0) {
          o_ << "(i" << non_matrix_dims << "__";
          if (matrix_dims > 1)
            o_ << ",i" << (non_matrix_dims + 1) << "__";
          o_ << ')';
        }
        o_ << ")) {" << EOL;
        generate_indent(indents_ + dims.size() + 1, o_);
        o_ << "std::stringstream msg__;" << EOL;
        generate_indent(indents_ + dims.size() + 1, o_);
        o_ << "msg__ << \"Undefined transformed parameter: "
           << name << "\"";
        for (size_t k = 0; k < dims.size(); ++k) {
          o_ << " << '['";
          o_ << " << i" << k << "__";
          o_ << " << ']'";
        }
        o_ << ';' << EOL;
        generate_indent(indents_ + dims.size() + 1, o_);
        o_ << "throw std::runtime_error(msg__.str());" << EOL;

        generate_indent(indents_ + dims.size(), o_);
        o_ << "}" << EOL;
        for (size_t k = 0; k < dims.size(); ++k) {
          generate_indent(indents_ + dims.size() - k - 1, o_);
          o_ << "}" << EOL;
        }
      }
    };

    void generate_validate_transformed_params(const std::vector<var_decl>& vs,
                                              int indent,
                                              std::ostream& o) {
      generate_comment("validate transformed parameters", indent, o);
      validate_transformed_params_visgen vis(indent, o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
      o << EOL;
    }

    struct idx_visgen : public visgen {
      explicit idx_visgen(std::ostream& o): visgen(o) { }
      void operator()(const uni_idx& i) const {
        o_ << "stan::model::index_uni(";
        generate_expression(i.idx_, o_);
        o_ << ")";
      }
      void operator()(const multi_idx& i) const {
        o_ << "stan::model::index_multi(";
        generate_expression(i.idxs_, o_);
        o_ << ")";
      }
      void operator()(const omni_idx& i) const {
        o_ << "stan::model::index_omni()";
      }
      void operator()(const lb_idx& i) const {
        o_ << "stan::model::index_min(";
        generate_expression(i.lb_, o_);
        o_ << ")";
      }
      void operator()(const ub_idx& i) const {
        o_ << "stan::model::index_max(";
        generate_expression(i.ub_, o_);
        o_ << ")";
      }
      void operator()(const lub_idx& i) const {
        o_ << "stan::model::index_min_max(";
        generate_expression(i.lb_, o_);
        o_ << ", ";
        generate_expression(i.ub_, o_);
        o_ << ")";
      }
    };

    void generate_idx(const idx& i, std::ostream& o) {
      idx_visgen vis(o);
      boost::apply_visitor(vis, i.idx_);
    }

    void generate_idxs(size_t pos, const std::vector<idx>& idxs,
                       std::ostream& o) {
      if (pos == idxs.size()) {
        o << "stan::model::nil_index_list()";
      } else {
        o << "stan::model::cons_list(";
        generate_idx(idxs[pos], o);
        o << ", ";
        generate_idxs(pos + 1, idxs, o);
        o << ")";
      }
    }

    void generate_idxs(const std::vector<idx>& idxs, std::ostream& o) {
      generate_idxs(0, idxs, o);
    }


    struct idx_user_visgen : public visgen {
      explicit idx_user_visgen(std::ostream& o): visgen(o) { }
      void operator()(const uni_idx& i) const {
        generate_expression(i.idx_, true, o_);
      }
      void operator()(const multi_idx& i) const {
        generate_expression(i.idxs_, true, o_);
      }
      void operator()(const omni_idx& i) const {
        o_ << " ";
      }
      void operator()(const lb_idx& i) const {
        generate_expression(i.lb_, true, o_);
        o_ << ": ";
      }
      void operator()(const ub_idx& i) const {
        o_ << " :";
        generate_expression(i.ub_, true, o_);
      }
      void operator()(const lub_idx& i) const {
        generate_expression(i.lb_, true, o_);
        o_ << ":";
        generate_expression(i.ub_, true, o_);
      }
    };

    void generate_idx_user(const idx& i, std::ostream& o) {
      idx_user_visgen vis(o);
      boost::apply_visitor(vis, i.idx_);
    }

    void generate_idxs_user(const std::vector<idx>& idxs, std::ostream& o) {
      if (idxs.size() == 0)
        return;
      o << "[";
      for (size_t i = 0; i < idxs.size(); ++i) {
        if (i > 0)
          o << ", ";
        generate_idx_user(idxs[i], o);
      }
      o << "]";
    }

    struct statement_visgen : public visgen {
      size_t indent_;
      bool include_sampling_;
      bool is_var_context_;
      bool is_fun_return_;
      statement_visgen(size_t indent,
                       bool include_sampling,
                       bool is_var_context,
                       bool is_fun_return,
                       std::ostream& o)
        : visgen(o),
          indent_(indent),
          include_sampling_(include_sampling),
          is_var_context_(is_var_context),
          is_fun_return_(is_fun_return) {
      }
      void operator()(const nil& /*x*/) const {
      }
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
      void operator()(const no_op_statement& /*x*/) const {
      }
    };


    struct is_numbered_statement_vis : public boost::static_visitor<bool> {
      bool operator()(const nil& st) const { return false; }
      bool operator()(const assignment& st) const { return true; }
      bool operator()(const assgn& st) const { return true; }
      bool operator()(const sample& st) const { return true; }
      bool operator()(const increment_log_prob_statement& t) const {
        return true;
      }
      bool operator()(const expression& st) const  { return true; }
      bool operator()(const statements& st) const  { return false; }
      bool operator()(const for_statement& st) const  { return true; }
      bool operator()(const conditional_statement& st) const { return true; }
      bool operator()(const while_statement& st) const { return true; }
      bool operator()(const break_continue_statement& st) const {
        return true;
      }
      bool operator()(const print_statement& st) const { return true; }
      bool operator()(const reject_statement& st) const { return true; }
      bool operator()(const no_op_statement& st) const { return true; }
      bool operator()(const return_statement& st) const { return true; }
    };

    void generate_statement(const statement& s,
                            int indent,
                            std::ostream& o,
                            bool include_sampling,
                            bool is_var_context,
                            bool is_fun_return) {
      is_numbered_statement_vis vis_is_numbered;
      if (boost::apply_visitor(vis_is_numbered, s.statement_)) {
        generate_indent(indent, o);
        o << "current_statement_begin__ = " <<  s.begin_line_ << ";"
          << EOL;
      }
      statement_visgen vis(indent, include_sampling, is_var_context,
                           is_fun_return, o);
      boost::apply_visitor(vis, s.statement_);
    }

    // FIXME:  don't ever call this -- call generate statement instead
    void generate_statements(const std::vector<statement>& ss,
                             int indent,
                             std::ostream& o,
                             bool include_sampling,
                             bool is_var_context,
                             bool is_fun_return) {
      statement_visgen vis(indent, include_sampling, is_var_context,
                           is_fun_return, o);
      for (size_t i = 0; i < ss.size(); ++i)
        boost::apply_visitor(vis, ss[i].statement_);
    }

    void generate_try(int indent,
                      std::ostream& o) {
      generate_indent(indent, o);
      o << "try {"
        << EOL;
    }

    void generate_catch_throw_located(int indent,
                                      std::ostream& o) {
      generate_indent(indent, o);
      o << "} catch (const std::exception& e) {"
        << EOL;
      generate_indent(indent + 1, o);
      o << "stan::lang::rethrow_located(e,current_statement_begin__);"
        << EOL;
      generate_comment("Next line prevents compiler griping about no return",
                       indent + 1, o);
      generate_indent(indent + 1, o);
      o << "throw std::runtime_error"
        << "(\"*** IF YOU SEE THIS, PLEASE REPORT A BUG ***\");"
        << EOL;
      generate_indent(indent, o);
      o << "}"
        << EOL;
    }

    void generate_located_statement(const statement& s,
                                    int indent,
                                    std::ostream& o,
                                    bool include_sampling,
                                    bool is_var_context,
                                    bool is_fun_return) {
      generate_try(indent, o);
      generate_statement(s, indent+1, o, include_sampling,
                         is_var_context, is_fun_return);
      generate_catch_throw_located(indent, o);
    }

    void generate_located_statements(const std::vector<statement>& ss,
                                     int indent,
                                     std::ostream& o,
                                     bool include_sampling,
                                     bool is_var_context,
                                     bool is_fun_return) {
      generate_try(indent, o);
      for (size_t i = 0; i < ss.size(); ++i)
        generate_statement(ss[i], indent + 1, o, include_sampling,
                           is_var_context, is_fun_return);
      generate_catch_throw_located(indent, o);
    }



    void generate_log_prob(const program& p,
                           std::ostream& o) {
      o << EOL;
      o << INDENT << "template <bool propto__, bool jacobian__, typename T__>"
        << EOL;
      o << INDENT << "T__ log_prob(vector<T__>& params_r__,"
        << EOL;
      o << INDENT << "             vector<int>& params_i__,"
        << EOL;
      o << INDENT << "             std::ostream* pstream__ = 0) const {"
        << EOL2;

      // use this dummy for inits
      o << INDENT2
        << "T__ DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      o << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;

      o << INDENT2 << "T__ lp__(0.0);"
        << EOL;
      o << INDENT2 << "stan::math::accumulator<T__> lp_accum__;"
        << EOL2;

      bool is_var_context = true;
      bool is_fun_return = false;

      generate_comment("model parameters", 2, o);
      generate_local_var_inits(p.parameter_decl_, is_var_context, true, o);
      o << EOL;

      generate_comment("transformed parameters", 2, o);
      generate_local_var_decls(p.derived_decl_.first, 2, o, is_var_context,
                               is_fun_return);
      o << EOL;

      bool include_sampling = true;
      generate_located_statements(p.derived_decl_.second, 2, o,
                                  include_sampling, is_var_context,
                                  is_fun_return);
      o << EOL;

      generate_validate_transformed_params(p.derived_decl_.first, 2, o);
      o << INDENT2
        << "const char* function__ = \"validate transformed params\";"
        << EOL;
      o << INDENT2
        << "(void) function__;  // dummy to suppress unused var warning"
        << EOL;

      generate_validate_var_decls(p.derived_decl_.first, 2, o);

      o << EOL;
      generate_comment("model body", 2, o);


      generate_located_statement(p.statement_, 2, o, include_sampling,
                                 is_var_context, is_fun_return);


      o << EOL;
      o << INDENT2 << "lp_accum__.add(lp__);" << EOL;
      o << INDENT2 << "return lp_accum__.sum();" << EOL2;
      o << INDENT << "} // log_prob()" << EOL2;

      o << INDENT
        << "template <bool propto, bool jacobian, typename T_>" << EOL;
      o << INDENT
        << "T_ log_prob(Eigen::Matrix<T_,Eigen::Dynamic,1>& params_r," << EOL;
      o << INDENT << "           std::ostream* pstream = 0) const {" << EOL;
      o << INDENT << "  std::vector<T_> vec_params_r;" << EOL;
      o << INDENT << "  vec_params_r.reserve(params_r.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < params_r.size(); ++i)" << EOL;
      o << INDENT << "    vec_params_r.push_back(params_r(i));" << EOL;
      o << INDENT << "  std::vector<int> vec_params_i;" << EOL;
      o << INDENT
        << "  return log_prob<propto,jacobian,T_>(vec_params_r, "
        << "vec_params_i, pstream);" << EOL;
      o << INDENT << "}" << EOL2;
    }

    struct dump_member_var_visgen : public visgen {
      var_resizing_visgen var_resizer_;
      var_size_validating_visgen var_size_validator_;
      explicit dump_member_var_visgen(std::ostream& o)
        : visgen(o),
          var_resizer_(var_resizing_visgen(o)),
          var_size_validator_(var_size_validating_visgen(o,
                                                    "data initialization")) {
      }
      void operator()(const nil& /*x*/) const { }  // dummy
      void operator()(const int_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_i__ = context__.vals_i(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        size_t indentation = 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_"
             << dim << "__ < " << x.name_ << "_limit_" << dim
             << "__; ++i_" << dim << "__) {" << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << " = vals_i__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 1 - dim, o_);
          o_ << "}" << EOL;
        }
      }
      // minor changes to int_var_decl
      void operator()(const double_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        size_t indentation = 1;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 1 - dim, o_);
          o_ << "}" << EOL;
        }
      }
      // extra outer loop around double_var_decl
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_resizer_(x);
        var_size_validator_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.M_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT2 << "}" << EOL;
      }
      // change variable name from vector_var_decl
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.N_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT2 << "}" << EOL;
      }
      // same as simplex
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT2 << "}" << EOL;
      }
      // diff name of dims from vector
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT2 << "}" << EOL;
      }
      // same as simplex
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT2 << "}" << EOL;
      }
      // same as simplex
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_i_vec_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "i_vec__ = 0; " << "i_vec__ < "
           << x.name_ << "_i_vec_lim__; ++i_vec__) {" << EOL;
        size_t indentation = 2;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "[i_vec__]";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT2 << "}" << EOL;
      }
      // extra loop and different accessor vs. vector
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\""
           << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_m_mat_lim__ = ";
        generate_expression(x.M_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_n_mat_lim__ = ";
        generate_expression(x.N_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
      void operator()(const corr_matrix_var_decl& x) const {
        // FIXME: cut-and-paste of cov_matrix,
        //        very slightly different from matrix
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_k_mat_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2
           << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < " << x.name_
           << "_k_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3
           << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < " << x.name_
           << "_k_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        // FIXME: cut and paste of cov_matrix
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\""
           << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;

        o_ << INDENT2 << "size_t " << x.name_ << "_m_mat_lim__ = ";
        generate_expression(x.M_, o_);
        o_ << ";" << EOL;

        o_ << INDENT2 << "size_t " << x.name_ << "_n_mat_lim__ = ";
        generate_expression(x.N_, o_);
        o_ << ";" << EOL;

        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;

        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }

        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        // FIXME: cut and paste of cholesky_factor_var_decl
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\""
           << x.name_ << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;

        o_ << INDENT2 << "size_t " << x.name_ << "_m_mat_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;

        o_ << INDENT2 << "size_t " << x.name_ << "_n_mat_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;

        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_n_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_m_mat_lim__; ++m_mat__) {" << EOL;

        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }

        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> dims = x.dims_;
        var_size_validator_(x);
        var_resizer_(x);
        o_ << INDENT2 << "vals_r__ = context__.vals_r(\"" << x.name_ << "\");"
           << EOL;
        o_ << INDENT2 << "pos__ = 0;" << EOL;
        o_ << INDENT2 << "size_t " << x.name_ << "_k_mat_lim__ = ";
        generate_expression(x.K_, o_);
        o_ << ";" << EOL;
        o_ << INDENT2 << "for (size_t " << "n_mat__ = 0; " << "n_mat__ < "
           << x.name_ << "_k_mat_lim__; ++n_mat__) {" << EOL;
        o_ << INDENT3 << "for (size_t " << "m_mat__ = 0; " << "m_mat__ < "
           << x.name_ << "_k_mat_lim__; ++m_mat__) {" << EOL;
        size_t indentation = 3;
        for (size_t dim_up = 0U; dim_up < dims.size(); ++dim_up) {
          size_t dim = dims.size() - dim_up - 1U;
          ++indentation;
          generate_indent(indentation, o_);
          o_ << "size_t " << x.name_ << "_limit_" << dim << "__ = ";
          generate_expression(dims[dim], o_);
          o_ << ";" << EOL;
          generate_indent(indentation, o_);
          o_ << "for (size_t i_" << dim << "__ = 0; i_" << dim << "__ < "
             << x.name_ << "_limit_" << dim << "__; ++i_" << dim << "__) {"
             << EOL;
        }
        generate_indent(indentation+1, o_);
        o_ << x.name_;
        for (size_t dim = 0; dim < dims.size(); ++dim)
          o_ << "[i_" << dim << "__]";
        o_ << "(m_mat__,n_mat__)";
        o_ << " = vals_r__[pos__++];" << EOL;
        for (size_t dim = 0; dim < dims.size(); ++dim) {
          generate_indent(dims.size() + 2 - dim, o_);
          o_ << "}" << EOL;
        }
        o_ << INDENT3 << "}" << EOL;
        o_ << INDENT2 << "}" << EOL;
      }
    };

    void suppress_warning(const std::string& indent,
                          const std::string& var_name,
                          std::ostream& o) {
      o << indent << "(void) "
        << var_name << ";"
        << " // dummy call to supress warning"
        << EOL;
    }

    void generate_member_var_inits(const std::vector<var_decl>& vs,
                                   std::ostream& o) {
      dump_member_var_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);
    }

    void generate_destructor(const std::string& model_name,
                             std::ostream& o) {
      o << EOL
        << INDENT << "~" << model_name << "() { }"
        << EOL2;
    }

    // know all data is set and range expressions only depend on data
    struct set_param_ranges_visgen : public visgen {
      explicit set_param_ranges_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const { }
      void operator()(const int_var_decl& x) const {
        generate_increment_i(x.dims_);
        // for loop for ranges
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_indent(i + 2, o_);
          o_ << "for (size_t i_" << i << "__ = 0; ";
          o_ << "i_" << i << "__ < ";
          generate_expression(x.dims_[i], o_);
          o_ << "; ++i_" << i << "__) {" << EOL;
        }
        // add range
        generate_indent(x.dims_.size() + 2, o_);
        o_ << "param_ranges_i__.push_back(std::pair<int, int>(";
        generate_expression(x.range_.low_, o_);
        o_ << ", ";
        generate_expression(x.range_.high_, o_);
        o_ << "));" << EOL;
        // close for loop
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          generate_indent(x.dims_.size() + 1 - i, o_);
          o_ << "}" << EOL;
        }
      }
      void operator()(const double_var_decl& x) const {
        generate_increment(x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        generate_increment(x.M_, x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        generate_increment(x.N_, x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        generate_increment(x.M_, x.N_, x.dims_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        o_ << INDENT2 << "num_params_r__ += (";
        generate_expression(x.K_, o_);
        o_ << ")";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const simplex_var_decl& x) const {
        // only K-1 vals
        o_ << INDENT2 << "num_params_r__ += (";
        generate_expression(x.K_, o_);
        o_ << " - 1)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const ordered_var_decl& x) const {
        generate_increment(x.K_, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        generate_increment(x.K_, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        o_ << INDENT2 << "num_params_r__ += ((";
        // N * (N + 1) / 2  +  (M - N) * M
        generate_expression(x.N_, o_);
        o_ << " * (";
        generate_expression(x.N_, o_);
        o_ << " + 1)) / 2 + (";
        generate_expression(x.M_, o_);
        o_ << " - ";
        generate_expression(x.N_, o_);
        o_ << ") * ";
        generate_expression(x.N_, o_);
        o_ << ")";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        // FIXME: cut and paste ofcorr_matrix_var_decl
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_, o_);
        o_ << " * (";
        generate_expression(x.K_, o_);
        o_ << " - 1)) / 2)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const cov_matrix_var_decl& x) const {
        // (K * (K - 1))/2 + K  ?? define fun(K) = ??
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_, o_);
        o_ << " * (";
        generate_expression(x.K_, o_);
        o_ << " - 1)) / 2 + ";
        generate_expression(x.K_, o_);
        o_ << ")";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      void operator()(const corr_matrix_var_decl& x) const {
        o_ << INDENT2 << "num_params_r__ += ((";
        generate_expression(x.K_, o_);
        o_ << " * (";
        generate_expression(x.K_, o_);
        o_ << " - 1)) / 2)";
        for (size_t i = 0; i < x.dims_.size(); ++i) {
          o_ << " * ";
          generate_expression(x.dims_[i], o_);
        }
        o_ << ";" << EOL;
      }
      // cut-and-paste from next for r
      void generate_increment_i(std::vector<expression> dims) const {
        if (dims.size() == 0) {
          o_ << INDENT2 << "++num_params_i__;" << EOL;
          return;
        }
        o_ << INDENT2 << "num_params_r__ += ";
        for (size_t i = 0; i < dims.size(); ++i) {
          if (i > 0) o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }
      void generate_increment(std::vector<expression> dims) const {
        if (dims.size() == 0) {
          o_ << INDENT2 << "++num_params_r__;" << EOL;
          return;
        }
        o_ << INDENT2 << "num_params_r__ += ";
        for (size_t i = 0; i < dims.size(); ++i) {
          if (i > 0) o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }
      void generate_increment(expression K,
                              std::vector<expression> dims) const {
        o_ << INDENT2 << "num_params_r__ += ";
        generate_expression(K, o_);
        for (size_t i = 0; i < dims.size(); ++i) {
          o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }
      void generate_increment(expression M, expression N,
                              std::vector<expression> dims) const {
        o_ << INDENT2 << "num_params_r__ += ";
        generate_expression(M, o_);
        o_ << " * ";
        generate_expression(N, o_);
        for (size_t i = 0; i < dims.size(); ++i) {
          o_ << " * ";
          generate_expression(dims[i], o_);
        }
        o_ << ";" << EOL;
      }
    };

    void generate_set_param_ranges(const std::vector<var_decl>& var_decls,
                                   std::ostream& o) {
      o << INDENT2 << "num_params_r__ = 0U;" << EOL;
      o << INDENT2 << "param_ranges_i__.clear();" << EOL;
      set_param_ranges_visgen vis(o);
      for (size_t i = 0; i < var_decls.size(); ++i)
        boost::apply_visitor(vis, var_decls[i].decl_);
    }

    void generate_constructor(const program& prog,
                              const std::string& model_name,
                              std::ostream& o) {
      // constructor without RNG or template parameter
      // FIXME(carpenter): remove this and only call full ctor
      o << INDENT << model_name << "(stan::io::var_context& context__," << EOL;
      o << INDENT << "    std::ostream* pstream__ = 0)" << EOL;
      o << INDENT2 << ": prob_grad(0) {" << EOL;
      o << INDENT2 << "typedef boost::ecuyer1988 rng_t;" << EOL;
      o << INDENT2 << "rng_t base_rng(0);  // 0 seed default" << EOL;
      o << INDENT2 << "ctor_body(context__, base_rng, pstream__);" << EOL;
      o << INDENT << "}" << EOL2;

      // constructor with specified RNG
      o << INDENT << "template <class RNG>" << EOL;
      o << INDENT << model_name << "(stan::io::var_context& context__," << EOL;
      o << INDENT << "    RNG& base_rng__," << EOL;
      o << INDENT << "    std::ostream* pstream__ = 0)" << EOL;
      o << INDENT2 << ": prob_grad(0) {" << EOL;
      o << INDENT2 << "ctor_body(context__, base_rng__, pstream__);" << EOL;
      o << INDENT << "}" << EOL2;

      // body of constructor now in function
      o << INDENT << "template <class RNG>" << EOL;
      o << INDENT << "void ctor_body(stan::io::var_context& context__," << EOL;
      o << INDENT << "               RNG& base_rng__," << EOL;
      o << INDENT << "               std::ostream* pstream__) {" << EOL;
      o << INDENT2 << "current_statement_begin__ = -1;" << EOL2;
      o << INDENT2 << "static const char* function__ = \""
        << model_name << "_namespace::" << model_name << "\";" << EOL;
      suppress_warning(INDENT2, "function__", o);
      o << INDENT2 << "size_t pos__;" << EOL;
      suppress_warning(INDENT2, "pos__", o);
      o << INDENT2 << "std::vector<int> vals_i__;" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;

      o << INDENT2
        << "double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      o << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;

      o << INDENT2 << "// initialize member variables" << EOL;
      generate_member_var_inits(prog.data_decl_, o);

      o << EOL;
      generate_comment("validate, data variables", 2, o);
      generate_validate_var_decls(prog.data_decl_, 2, o);
      generate_comment("initialize data variables", 2, o);
      generate_var_resizing(prog.derived_data_decl_.first, o);
      o << EOL;

      bool include_sampling = false;
      bool is_var_context = false;
      bool is_fun_return = false;

      // need to fix generate_located_statements
      generate_located_statements(prog.derived_data_decl_.second,
                                  2, o, include_sampling, is_var_context,
                                  is_fun_return);

      o << EOL;
      generate_comment("validate transformed data", 2, o);
      generate_validate_var_decls(prog.derived_data_decl_.first, 2, o);

      o << EOL;
      generate_comment("set parameter ranges", 2, o);
      generate_set_param_ranges(prog.parameter_decl_, o);
      // o << EOL << INDENT2 << "set_param_ranges();" << EOL;

      o << INDENT << "}" << EOL;
    }

    struct generate_init_visgen : public visgen {
      var_size_validating_visgen var_size_validator_;
      explicit generate_init_visgen(std::ostream& o)
        : visgen(o),
          var_size_validator_(o, "initialization") {
      }
      void operator()(const nil& /*x*/) const { }  // dummy
      void operator()(const int_var_decl& x) const {
        generate_check_int(x.name_, x.dims_.size());
        var_size_validator_(x);
        generate_declaration(x.name_, "int", x.dims_, nil(), nil(), x.def_);
        generate_buffer_loop("i", x.name_, x.dims_);
        generate_write_loop("integer(", x.name_, x.dims_);
      }
      template <typename D>
      std::string function_args(const std::string& fun_prefix,
                                const D& x) const {
        std::stringstream ss;
        ss << fun_prefix;
        if (has_lub(x)) {
          ss << "_lub_unconstrain(";
          generate_expression(x.range_.low_.expr_, ss);
          ss << ',';
          generate_expression(x.range_.high_.expr_, ss);
          ss << ',';
        } else if (has_lb(x)) {
          ss << "_lb_unconstrain(";
          generate_expression(x.range_.low_.expr_, ss);
          ss << ',';
        } else if (has_ub(x)) {
          ss << "_ub_unconstrain(";
          generate_expression(x.range_.high_.expr_, ss);
          ss << ',';
        } else {
          ss << "_unconstrain(";
        }
        return ss.str();
      }

      void operator()(const double_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size());
        var_size_validator_(x);
        generate_declaration(x.name_, "double", x.dims_, nil(), nil(), x.def_);
        if (is_nil(x.def_)) {
          generate_buffer_loop("r", x.name_, x.dims_);
        }
        generate_write_loop(function_args("scalar", x),
                            x.name_, x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.M_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.M_);
        generate_write_loop(function_args("vector", x),
                            x.name_, x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "row_vector_d", x.dims_, x.N_, nil(),
                             x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.N_);
        generate_write_loop(function_args("row_vector", x),
                            x.name_, x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 2);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.M_, x.N_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.M_, x.N_);
        generate_write_loop(function_args("matrix", x),
                            x.name_, x.dims_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("unit_vector_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("simplex_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const ordered_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("ordered_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 1);
        var_size_validator_(x);
        generate_declaration(x.name_, "vector_d", x.dims_, x.K_, nil(), x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_);
        generate_write_loop("positive_ordered_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 2);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.M_, x.N_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.M_, x.N_);
        generate_write_loop("cholesky_factor_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 2);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.K_, x.K_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_, x.K_);
        generate_write_loop("cholesky_corr_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 2);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.K_, x.K_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_, x.K_);
        generate_write_loop("cov_matrix_unconstrain(", x.name_, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        generate_check_double(x.name_, x.dims_.size() + 2);
        var_size_validator_(x);
        generate_declaration(x.name_, "matrix_d", x.dims_, x.K_, x.K_, x.def_);
        generate_buffer_loop("r", x.name_, x.dims_, x.K_, x.K_);
        generate_write_loop("corr_matrix_unconstrain(", x.name_, x.dims_);
      }
      void generate_write_loop(const std::string& write_method_name,
                               const std::string& var_name,
                               const std::vector<expression>& dims) const {
        generate_dims_loop_fwd(dims);
        o_ << "try {"
           << EOL
           << INDENT3
           << "writer__." << write_method_name;
        generate_name_dims(var_name, dims.size());
        o_ << ");"
           << EOL
           << INDENT2
           << "} catch (const std::exception& e) { "
           << EOL
           << INDENT3
           << "throw std::runtime_error("
           << "std::string(\"Error transforming variable "
           << var_name << ": \") + e.what());"
           << EOL
           << INDENT2
           << "}"
           << EOL;
      }
      void generate_name_dims(const std::string name,
                              size_t num_dims) const {
        o_ << name;
        for (size_t i = 0; i < num_dims; ++i)
          o_ << "[i" << i << "__]";
      }
      void generate_declaration(const std::string& name,
                                const std::string& base_type,
                                const std::vector<expression>& dims,
                                const expression& type_arg1 = expression(),
                                const expression& type_arg2 = expression(),
                                const expression& definition = expression())
      const {
        o_ << INDENT2 << "// generate_declaration " << name << std::endl;
        o_ << INDENT2;
        generate_type(base_type, dims, dims.size(), o_);
        o_ << ' ' << name;
        generate_initializer(o_, base_type, dims, type_arg1, type_arg2);
      }
      void generate_indent_num_dims(size_t base_indent,
                                    const std::vector<expression>& dims,
                                    const expression& dim1,
                                    const expression& dim2) const {
        generate_indent(dims.size() + base_indent, o_);
        if (!is_nil(dim1)) o_ << INDENT;
        if (!is_nil(dim2)) o_ << INDENT;
      }
      void generate_buffer_loop(const std::string& base_type,
                                const std::string& name,
                                const std::vector<expression>& dims,
                                const expression& dim1 = expression(),
                                const expression& dim2 = expression(),
                                int indent = 2U) const {
        size_t size = dims.size();
        bool is_matrix = !is_nil(dim1) && !is_nil(dim2);
        bool is_vector = !is_nil(dim1) && is_nil(dim2);
        int extra_indent = is_matrix ? 2U : is_vector ? 1U : 0U;
        if (is_matrix) {
          generate_indent(indent, o_);
          o_ << "for (int j2__ = 0U; j2__ < ";
          generate_expression(dim2.expr_, o_);
          o_ << "; ++j2__)" << EOL;

          generate_indent(indent+1, o_);
          o_ << "for (int j1__ = 0U; j1__ < ";
          generate_expression(dim1.expr_, o_);
          o_ << "; ++j1__)" << EOL;
        } else if (is_vector) {
          generate_indent(indent, o_);
          o_ << "for (int j1__ = 0U; j1__ < ";
          generate_expression(dim1.expr_, o_);
          o_ << "; ++j1__)" << EOL;
        }
        for (size_t i = 0; i < size; ++i) {
          size_t idx = size - i - 1;
          generate_indent(i + indent + extra_indent, o_);
          o_ << "for (int i" << idx << "__ = 0U; i" << idx << "__ < ";
          generate_expression(dims[idx].expr_, o_);
          o_ << "; ++i" << idx << "__)" << EOL;
        }
        generate_indent_num_dims(2U, dims, dim1, dim2);
        o_ << name;
        for (size_t i = 0; i < dims.size(); ++i)
          o_ << "[i" << i << "__]";
        if (is_matrix)
          o_ << "(j1__,j2__)";
        else if (is_vector)
          o_ << "(j1__)";
        o_ << " = vals_" << base_type << "__[pos__++];" << EOL;
      }
      void generate_dims_loop_fwd(const std::vector<expression>& dims,
                                  int indent = 2U) const {
        size_t size = dims.size();
        for (size_t i = 0; i < size; ++i) {
          generate_indent(i + indent, o_);
          o_ << "for (int i" << i << "__ = 0U; i" << i << "__ < ";
          generate_expression(dims[i].expr_, o_);
          o_ << "; ++i" << i << "__)" << EOL;
        }
        generate_indent(2U + dims.size(), o_);
      }
      void generate_check_int(const std::string& name, size_t /*n*/) const {
        o_ << EOL << INDENT2
           << "if (!(context__.contains_i(\"" << name << "\")))"
           << EOL << INDENT3
           << "throw std::runtime_error(\"variable " << name << " missing\");"
           << EOL;
        o_ << INDENT2 << "vals_i__ = context__.vals_i(\"" << name << "\");"
           << EOL;
        o_ << INDENT2 << "pos__ = 0U;" << EOL;
      }
      void generate_check_double(const std::string& name, size_t /*n*/) const {
        o_ << EOL << INDENT2
           << "if (!(context__.contains_r(\"" << name << "\")))"
           << EOL << INDENT3
           << "throw std::runtime_error(\"variable " << name << " missing\");"
           << EOL;
        o_ << INDENT2
           << "vals_r__ = context__.vals_r(\"" << name << "\");" << EOL;
        o_ << INDENT2 << "pos__ = 0U;" << EOL;
      }
    };


    void generate_init_method(const std::vector<var_decl>& vs,
                              std::ostream& o) {
      o << EOL;
      o << INDENT
        << "void transform_inits(const stan::io::var_context& context__,"
        << EOL;
      o << INDENT << "                     std::vector<int>& params_i__,"
        << EOL;
      o << INDENT << "                     std::vector<double>& params_r__,"
        << EOL;
      o << INDENT << "                     std::ostream* pstream__) const {"
        << EOL;
      o << INDENT2 << "stan::io::writer<double> "
        << "writer__(params_r__,params_i__);"
        << EOL;
      o << INDENT2 << "size_t pos__;" << EOL;
      o << INDENT2 << "(void) pos__; // dummy call to supress warning" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;
      o << INDENT2 << "std::vector<int> vals_i__;"
        << EOL;
      generate_init_visgen vis(o);
      for (size_t i = 0; i < vs.size(); ++i)
        boost::apply_visitor(vis, vs[i].decl_);

      o << EOL
        << INDENT2 << "params_r__ = writer__.data_r();" << EOL;
      o << INDENT2 << "params_i__ = writer__.data_i();" << EOL;
      o << INDENT << "}" << EOL2;

      o << INDENT
        << "void transform_inits(const stan::io::var_context& context," << EOL;
      o << INDENT
        << "                     "
        << "Eigen::Matrix<double,Eigen::Dynamic,1>& params_r," << EOL;
      o << INDENT
        << "                     std::ostream* pstream__) const {" << EOL;
      o << INDENT << "  std::vector<double> params_r_vec;" << EOL;
      o << INDENT << "  std::vector<int> params_i_vec;" << EOL;
      o << INDENT
        << "  transform_inits(context, params_i_vec, params_r_vec, pstream__);"
        << EOL;
      o << INDENT << "  params_r.resize(params_r_vec.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < params_r.size(); ++i)" << EOL;
      o << INDENT << "    params_r(i) = params_r_vec[i];" << EOL;
      o << INDENT << "}" << EOL2;
    }

    struct write_dims_visgen : public visgen {
      explicit write_dims_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const  { }
      void operator()(const int_var_decl& x) const {
        generate_dims_array(EMPTY_EXP_VECTOR, x.dims_);
      }
      void operator()(const double_var_decl& x) const {
        generate_dims_array(EMPTY_EXP_VECTOR, x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.N_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_dims_array(matrix_args, x.dims_);
      }
      void
      generate_dims_array(const std::vector<expression>& matrix_dims_exprs,
                          const std::vector<expression>& array_dims_exprs)
        const {
        o_ << INDENT2 << "dims__.resize(0);" << EOL;
        for (size_t i = 0; i < array_dims_exprs.size(); ++i) {
          o_ << INDENT2 << "dims__.push_back(";
          generate_expression(array_dims_exprs[i].expr_, o_);
          o_ << ");" << EOL;
        }
        // cut and paste above with matrix_dims_exprs
        for (size_t i = 0; i < matrix_dims_exprs.size(); ++i) {
          o_ << INDENT2 << "dims__.push_back(";
          generate_expression(matrix_dims_exprs[i].expr_, o_);
          o_ << ");" << EOL;
        }
        o_ << INDENT2 << "dimss__.push_back(dims__);" << EOL;
      }
    };

    void generate_dims_method(const program& prog,
                              std::ostream& o) {
      write_dims_visgen vis(o);
      o << EOL << INDENT
        << "void get_dims(std::vector<std::vector<size_t> >& dimss__) const {"
        << EOL;

      o << INDENT2 << "dimss__.resize(0);" << EOL;
      o << INDENT2 << "std::vector<size_t> dims__;" << EOL;

      // parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i) {
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);
      }
      // transformed parameters
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.derived_decl_.first[i].decl_);
      }
      // generated quantities
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.generated_decl_.first[i].decl_);
      }
      o << INDENT << "}" << EOL2;
    }



    struct write_param_names_visgen : public visgen {
      explicit write_param_names_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const  { }
      void operator()(const int_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const double_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const vector_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const row_vector_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const matrix_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const simplex_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const ordered_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        generate_param_names(x.name_);
      }
      void
      generate_param_names(const std::string& name) const {
        o_ << INDENT2
           << "names__.push_back(\"" << name << "\");"
           << EOL;
      }
    };


    void generate_param_names_method(const program& prog,
                                     std::ostream& o) {
      write_param_names_visgen vis(o);
      o << EOL << INDENT
        << "void get_param_names(std::vector<std::string>& names__) const {"
        << EOL;

      o << INDENT2
        << "names__.resize(0);"
        << EOL;

      // parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i) {
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);
      }
      // transformed parameters
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.derived_decl_.first[i].decl_);
      }
      // generated quantities
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.generated_decl_.first[i].decl_);
      }

      o << INDENT << "}" << EOL2;
    }



    struct constrained_param_names_visgen : public visgen {
      explicit constrained_param_names_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const  { }
      void operator()(const int_var_decl& x) const {
        generate_param_names_array(EMPTY_EXP_VECTOR, x.name_, x.dims_);
      }
      void operator()(const double_var_decl& x) const {
        generate_param_names_array(EMPTY_EXP_VECTOR, x.name_, x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.N_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void
      generate_param_names_array(const std::vector<expression>& matrix_dims,
                                 const std::string& name,
                                 const std::vector<expression>& dims) const {
        // begin for loop dims
        std::vector<expression> combo_dims(dims);
        for (size_t i = 0; i < matrix_dims.size(); ++i)
          combo_dims.push_back(matrix_dims[i]);

        for (size_t i = combo_dims.size(); i-- > 0; ) {
          generate_indent(1 + combo_dims.size() - i, o_);
          o_ << "for (int k_" << i << "__ = 1;"
             << " k_" << i << "__ <= ";
          generate_expression(combo_dims[i].expr_, o_);
          o_ << "; ++k_" << i << "__) {" << EOL;  // begin (1)
        }

        generate_indent(2 + combo_dims.size(), o_);
        o_ << "param_name_stream__.str(std::string());" << EOL;

        generate_indent(2 + combo_dims.size(), o_);
        o_ << "param_name_stream__ << \"" << name << '"';

        for (size_t i = 0; i < combo_dims.size(); ++i)
          o_ << " << '.' << k_" << i << "__";
        o_ << ';' << EOL;

        generate_indent(2 + combo_dims.size(), o_);
        o_ << "param_names__.push_back(param_name_stream__.str());" << EOL;

        // end for loop dims
        for (size_t i = 0; i < combo_dims.size(); ++i) {
          generate_indent(1 + combo_dims.size() - i, o_);
          o_ << "}" << EOL;  // end (1)
        }
      }
    };


    void generate_constrained_param_names_method(const program& prog,
                                                 std::ostream& o) {
      o << EOL << INDENT
        << "void constrained_param_names("
        << "std::vector<std::string>& param_names__,"
        << EOL << INDENT
        << "                             bool include_tparams__ = true,"
        << EOL << INDENT
        << "                             bool include_gqs__ = true) const {"
        << EOL << INDENT2
        << "std::stringstream param_name_stream__;" << EOL;

      constrained_param_names_visgen vis(o);
      // parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i) {
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);
      }

      o << EOL << INDENT2
        << "if (!include_gqs__ && !include_tparams__) return;"
        << EOL;

      // transformed parameters
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.derived_decl_.first[i].decl_);
      }

      o << EOL << INDENT2
        << "if (!include_gqs__) return;"
        << EOL;

      // generated quantities
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.generated_decl_.first[i].decl_);
      }

      o << INDENT << "}" << EOL2;
    }

    struct unconstrained_param_names_visgen : public visgen {
      explicit unconstrained_param_names_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const  { }
      void operator()(const int_var_decl& x) const {
        generate_param_names_array(EMPTY_EXP_VECTOR, x.name_, x.dims_);
      }
      void operator()(const double_var_decl& x) const {
        generate_param_names_array(EMPTY_EXP_VECTOR, x.name_, x.dims_);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.N_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.M_);
        matrix_args.push_back(x.N_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(binary_op(x.K_, "-", int_literal(1)));
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(x.K_);
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        // FIXME: cut-and-paste of cov_matrix
        std::vector<expression> matrix_args;
        // (N * (N + 1)) / 2 + (M - N) * N
        matrix_args.push_back(binary_op(binary_op(binary_op(x.N_,
                                                            "*",
                                                            binary_op(x.N_,
                                                                      "+",
                                                               int_literal(1))),
                                                  "/",
                                                  int_literal(2)),
                                        "+",
                                        binary_op(binary_op(x.M_,
                                                            "-",
                                                            x.N_),
                                                  "*",
                                                  x.N_)));
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        // FIXME: cut-and-paste of corr_matrix
        std::vector<expression> matrix_args;
        // (K * (K - 1)) / 2
        matrix_args.push_back(binary_op(binary_op(x.K_,
                                                  "*",
                                                  binary_op(x.K_,
                                                            "-",
                                                            int_literal(1))),
                                        "/",
                                        int_literal(2)));
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(binary_op(x.K_,
                                        "+",
                                        binary_op(binary_op(x.K_,
                                                            "*",
                                                            binary_op(x.K_,
                                                                      "-",
                                                               int_literal(1))),
                               "/",
                               int_literal(2))));
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matrix_args;
        matrix_args.push_back(binary_op(binary_op(x.K_,
                                                  "*",
                                                  binary_op(x.K_,
                                                            "-",
                                                            int_literal(1))),
                                        "/",
                                        int_literal(2)));
        generate_param_names_array(matrix_args, x.name_, x.dims_);
      }
      // FIXME: sharing instead of cut-and-paste from constrained
      void
      generate_param_names_array(const std::vector<expression>& matrix_dims,
                                 const std::string& name,
                                 const std::vector<expression>& dims) const {
        // begin for loop dims
        std::vector<expression> combo_dims(dims);
        for (size_t i = 0; i < matrix_dims.size(); ++i)
          combo_dims.push_back(matrix_dims[i]);

        for (size_t i = combo_dims.size(); i-- > 0; ) {
          generate_indent(1 + combo_dims.size() - i, o_);
          o_ << "for (int k_" << i << "__ = 1;"
             << " k_" << i << "__ <= ";
          generate_expression(combo_dims[i].expr_, o_);
          o_ << "; ++k_" << i << "__) {" << EOL;  // begin (1)
        }

        generate_indent(2 + combo_dims.size(), o_);
        o_ << "param_name_stream__.str(std::string());" << EOL;

        generate_indent(2 + combo_dims.size(), o_);
        o_ << "param_name_stream__ << \"" << name << '"';

        for (size_t i = 0; i < combo_dims.size(); ++i)
          o_ << " << '.' << k_" << i << "__";
        o_ << ';' << EOL;

        generate_indent(2 + combo_dims.size(), o_);
        o_ << "param_names__.push_back(param_name_stream__.str());" << EOL;

        // end for loop dims
        for (size_t i = 0; i < combo_dims.size(); ++i) {
          generate_indent(1 + combo_dims.size() - i, o_);
          o_ << "}" << EOL;  // end (1)
        }
      }
    };


    void generate_unconstrained_param_names_method(const program& prog,
                                                   std::ostream& o) {
      o << EOL << INDENT
        << "void unconstrained_param_names("
        << "std::vector<std::string>& param_names__,"
        << EOL << INDENT
        << "                               bool include_tparams__ = true,"
        << EOL << INDENT
        << "                               bool include_gqs__ = true) const {"
        << EOL << INDENT2
        << "std::stringstream param_name_stream__;" << EOL;

      unconstrained_param_names_visgen vis(o);
      // parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i) {
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);
      }

      o << EOL << INDENT2
        << "if (!include_gqs__ && !include_tparams__) return;"
        << EOL;

      // transformed parameters
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.derived_decl_.first[i].decl_);
      }

      o << EOL << INDENT2
        << "if (!include_gqs__) return;"
        << EOL;

      // generated quantities
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i) {
        boost::apply_visitor(vis, prog.generated_decl_.first[i].decl_);
      }

      o << INDENT << "}" << EOL2;
    }


    // see init_member_var_visgen for cut & paste
    struct write_array_visgen : public visgen {
      explicit write_array_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const { }
      void operator()(const int_var_decl& x) const {
        generate_initialize_array("int", "integer", EMPTY_EXP_VECTOR,
                                  x.name_, x.dims_);
      }
      // fixme -- reuse cut-and-pasted from other lub reader case
      template <typename D>
      void generate_initialize_array_bounded(const D& x,
                                             const std::string& base_type,
                                             const std::string& read_fun_prefix,
                                const std::vector<expression>& dim_args) const {
        std::vector<expression> read_args;
        std::string read_fun(read_fun_prefix);
        if (has_lub(x)) {
          read_fun += "_lub";
          read_args.push_back(x.range_.low_);
          read_args.push_back(x.range_.high_);
        } else if (has_lb(x)) {
          read_fun += "_lb";
          read_args.push_back(x.range_.low_);
        } else if (has_ub(x)) {
          read_fun += "_ub";
          read_args.push_back(x.range_.high_);
        }
        for (size_t i = 0; i < dim_args.size(); ++i)
          read_args.push_back(dim_args[i]);
        generate_initialize_array(base_type, read_fun, read_args,
                                  x.name_, x.dims_);
      }

      void operator()(const double_var_decl& x) const {
        std::vector<expression> read_args;
        generate_initialize_array_bounded(x, "double", "scalar", read_args);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        generate_initialize_array_bounded(x, "vector_d", "vector", read_args);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.N_);
        generate_initialize_array_bounded(x, "row_vector_d", "row_vector",
                                          read_args);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        read_args.push_back(x.N_);
        generate_initialize_array_bounded(x, "matrix_d", "matrix", read_args);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_d", "unit_vector", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_d", "simplex", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_d", "ordered", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("vector_d", "positive_ordered", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.M_);
        read_args.push_back(x.N_);
        generate_initialize_array("matrix_d", "cholesky_factor", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("matrix_d", "cholesky_corr", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("matrix_d", "cov_matrix", read_args,
                                  x.name_, x.dims_);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> read_args;
        read_args.push_back(x.K_);
        generate_initialize_array("matrix_d", "corr_matrix", read_args,
                                  x.name_, x.dims_);
      }
      void generate_initialize_array(const std::string& var_type,
                                     const std::string& read_type,
                                     const std::vector<expression>& read_args,
                                     const std::string& name,
                                     const std::vector<expression>& dims) const{
        if (dims.size() == 0) {
          generate_indent(2, o_);
          o_ << var_type << " ";
          o_ << name << " = in__." << read_type  << "_constrain(";
          for (size_t j = 0; j < read_args.size(); ++j) {
            if (j > 0) o_ << ",";
            generate_expression(read_args[j], o_);
          }
          o_ << ");" << EOL;
          return;
        }
        o_ << INDENT2;
        for (size_t i = 0; i < dims.size(); ++i) o_ << "vector<";
        o_ << var_type;
        for (size_t i = 0; i < dims.size(); ++i) o_ << "> ";
        o_ << name << ";" << EOL;
        std::string name_dims(name);
        for (size_t i = 0; i < dims.size(); ++i) {
          generate_indent(i + 2, o_);
          o_ << "size_t dim_"  << name << "_" << i << "__ = ";
          generate_expression(dims[i], o_);
          o_ << ";" << EOL;
          if (i < dims.size() - 1) {
            generate_indent(i + 2, o_);
            o_ << name_dims << ".resize(dim_" << name << "_" << i << "__);"
               << EOL;
            name_dims.append("[k_").append(to_string(i)).append("__]");
          }
          generate_indent(i + 2, o_);
          o_ << "for (size_t k_" << i << "__ = 0;"
             << " k_" << i << "__ < dim_" << name << "_" << i << "__;"
             << " ++k_" << i << "__) {" << EOL;
          if (i == dims.size() - 1) {
            generate_indent(i + 3, o_);
            o_ << name_dims << ".push_back(in__." << read_type << "_constrain(";
            for (size_t j = 0; j < read_args.size(); ++j) {
              if (j > 0) o_ << ",";
              generate_expression(read_args[j], o_);
            }
            o_ << "));" << EOL;
          }
        }

        for (size_t i = dims.size(); i > 0; --i) {
          generate_indent(i + 1, o_);
          o_ << "}" << EOL;
        }
      }
    };




    struct write_array_vars_visgen : public visgen {
      explicit write_array_vars_visgen(std::ostream& o)
        : visgen(o) {
      }
      void operator()(const nil& /*x*/) const { }
      // FIXME: template these out
      void operator()(const int_var_decl& x) const {
        write_array(x.name_, x.dims_, EMPTY_EXP_VECTOR);
      }
      void operator()(const double_var_decl& x) const {
        write_array(x.name_, x.dims_, EMPTY_EXP_VECTOR);
      }
      void operator()(const vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.M_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const row_vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.N_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const matrix_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.M_);
        matdims.push_back(x.N_);
        write_array(x.name_, x.dims_, matdims);
      }
      void operator()(const unit_vector_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const simplex_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const ordered_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const positive_ordered_var_decl& x) const {
        std::vector<expression> dims(x.dims_);
        dims.push_back(x.K_);
        write_array(x.name_, dims, EMPTY_EXP_VECTOR);
      }
      void operator()(const cholesky_factor_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.M_);
        matdims.push_back(x.N_);
        write_array(x.name_, x.dims_, matdims);
      }
      void operator()(const cholesky_corr_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.K_);
        matdims.push_back(x.K_);
        write_array(x.name_, x.dims_, matdims);
      }
      void operator()(const cov_matrix_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.K_);
        matdims.push_back(x.K_);
        write_array(x.name_, x.dims_, matdims);
      }
      void operator()(const corr_matrix_var_decl& x) const {
        std::vector<expression> matdims;
        matdims.push_back(x.K_);
        matdims.push_back(x.K_);
        write_array(x.name_, x.dims_, matdims);
      }
      void write_array(const std::string& name,
                       const std::vector<expression>& arraydims,
                       const std::vector<expression>& matdims) const {
        std::vector<expression> dims(arraydims);
        for (size_t i = 0; i < matdims.size(); ++i)
          dims.push_back(matdims[i]);

        if (dims.size() == 0) {
          o_ << INDENT2 << "vars__.push_back(" << name << ");" << EOL;
          return;
        }

        // for (size_t i = 0; i < dims.size(); ++i) {
        for (size_t i = dims.size(); i > 0; ) {
          --i;
          generate_indent((dims.size() - i) + 1, o_);
          o_ << "for (int k_" << i << "__ = 0;"
             << " k_" << i << "__ < ";
          generate_expression(dims[i], o_);
          o_ << "; ++k_" << i << "__) {" << EOL;
        }

        generate_indent(dims.size() + 2, o_);
        o_ << "vars__.push_back(" << name;
        if (arraydims.size() > 0) {
          o_ << '[';
          for (size_t i = 0; i < arraydims.size(); ++i) {
            if (i > 0) o_ << "][";
            o_ << "k_" << i << "__";
          }
          o_ << ']';
        }
        if (matdims.size() > 0) {
          o_ << "(k_" << arraydims.size() << "__";
          if (matdims.size() > 1)
            o_ << ", k_" << (arraydims.size() + 1) << "__";
          o_ << ")";
        }
        o_ << ");" << EOL;

        for (size_t i = dims.size(); i > 0; --i) {
          generate_indent(i + 1, o_);
          o_ << "}" << EOL;
        }
      }
    };


    void generate_write_array_method(const program& prog,
                                     const std::string& model_name,
                                     std::ostream& o) {
      o << INDENT << "template <typename RNG>" << EOL;
      o << INDENT << "void write_array(RNG& base_rng__," << EOL;
      o << INDENT << "                 std::vector<double>& params_r__," << EOL;
      o << INDENT << "                 std::vector<int>& params_i__," << EOL;
      o << INDENT << "                 std::vector<double>& vars__," << EOL;
      o << INDENT << "                 bool include_tparams__ = true," << EOL;
      o << INDENT << "                 bool include_gqs__ = true," << EOL;
      o << INDENT
        << "                 std::ostream* pstream__ = 0) const {" << EOL;
      o << INDENT2 << "vars__.resize(0);" << EOL;
      o << INDENT2
        << "stan::io::reader<double> in__(params_r__,params_i__);"<< EOL;
      o << INDENT2 << "static const char* function__ = \""
        << model_name << "_namespace::write_array\";" << EOL;
      suppress_warning(INDENT2, "function__", o);

      // declares, reads, and sets parameters
      generate_comment("read-transform, write parameters", 2, o);
      write_array_visgen vis(o);
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);

      // this is for all other values
      write_array_vars_visgen vis_writer(o);

      // writes parameters
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        boost::apply_visitor(vis_writer, prog.parameter_decl_[i].decl_);
      o << EOL;

      o << INDENT2 << "if (!include_tparams__) return;"
        << EOL;
      generate_comment("declare and define transformed parameters", 2, o);
      o << INDENT2 <<  "double lp__ = 0.0;" << EOL;
      suppress_warning(INDENT2, "lp__", o);
      o << INDENT2 << "stan::math::accumulator<double> lp_accum__;" << EOL2;

      o << INDENT2
        << "double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      o << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;

      bool is_var_context = false;
      bool is_fun_return = false;
      generate_local_var_decls(prog.derived_decl_.first, 2, o, is_var_context,
                               is_fun_return);
      o << EOL;
      bool include_sampling = false;
      generate_located_statements(prog.derived_decl_.second, 2, o,
                                  include_sampling, is_var_context,
                                  is_fun_return);
      o << EOL;

      generate_comment("validate transformed parameters", 2, o);
      generate_validate_var_decls(prog.derived_decl_.first, 2, o);
      o << EOL;

      generate_comment("write transformed parameters", 2, o);
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i)
        boost::apply_visitor(vis_writer, prog.derived_decl_.first[i].decl_);
      o << EOL;

      o << INDENT2 << "if (!include_gqs__) return;"
        << EOL;
      generate_comment("declare and define generated quantities", 2, o);
      generate_local_var_decls(prog.generated_decl_.first, 2, o,
                               is_var_context, is_fun_return);

      o << EOL;
      generate_located_statements(prog.generated_decl_.second, 2, o,
                                  include_sampling, is_var_context,
                                  is_fun_return);
      o << EOL;

      generate_comment("validate generated quantities", 2, o);
      generate_validate_var_decls(prog.generated_decl_.first, 2, o);
      o << EOL;

      generate_comment("write generated quantities", 2, o);
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i)
        boost::apply_visitor(vis_writer, prog.generated_decl_.first[i].decl_);
      if (prog.generated_decl_.first.size() > 0)
        o << EOL;

      o << INDENT << "}" << EOL2;

      o << INDENT << "template <typename RNG>" << EOL;
      o << INDENT << "void write_array(RNG& base_rng," << EOL;
      o << INDENT
        << "                 Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,"
        << EOL;
      o << INDENT
        << "                 Eigen::Matrix<double,Eigen::Dynamic,1>& vars,"
        << EOL;
      o << INDENT << "                 bool include_tparams = true," << EOL;
      o << INDENT << "                 bool include_gqs = true," << EOL;
      o << INDENT
        << "                 std::ostream* pstream = 0) const {" << EOL;
      o << INDENT
        << "  std::vector<double> params_r_vec(params_r.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < params_r.size(); ++i)" << EOL;
      o << INDENT << "    params_r_vec[i] = params_r(i);" << EOL;
      o << INDENT << "  std::vector<double> vars_vec;" << EOL;
      o << INDENT << "  std::vector<int> params_i_vec;" << EOL;
      o << INDENT
        << "  write_array(base_rng,params_r_vec,params_i_vec,"
        << "vars_vec,include_tparams,include_gqs,pstream);" << EOL;
      o << INDENT << "  vars.resize(vars_vec.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < vars.size(); ++i)" << EOL;
      o << INDENT << "    vars(i) = vars_vec[i];" << EOL;
      o << INDENT << "}" << EOL2;
    }

    void generate_model_name_method(const std::string& model_name,
                                    std::ostream& out) {
      out << INDENT << "static std::string model_name() {" << EOL
          << INDENT2 << "return \"" << model_name << "\";" << EOL
          << INDENT << "}" << EOL2;
    }

    void generate_model_typedef(const std::string& model_name,
                                std::ostream& out) {
      out << "typedef " << model_name << "_namespace::" << model_name
          << " stan_model;" <<EOL2;
    }

    void generate_bare_type(const expr_type& t,
                            const std::string& scalar_t_name,
                            std::ostream& out) {
      for (size_t d = 0; d < t.num_dims_; ++d)
        out << "std::vector<";

      bool is_template_type = false;
      switch (t.base_type_) {
      case INT_T :
        out << "int";
        is_template_type = false;
        break;
      case DOUBLE_T:
        out << scalar_t_name;
        is_template_type = false;
        break;
      case VECTOR_T:
        out << "Eigen::Matrix<"
            << scalar_t_name
            << ", Eigen::Dynamic,1>";
        is_template_type = true;
        break;
      case ROW_VECTOR_T:
        out << "Eigen::Matrix<"
            << scalar_t_name
            << ", 1,Eigen::Dynamic>";
        is_template_type = true;
        break;
      case MATRIX_T:
        out << "Eigen::Matrix<"
            << scalar_t_name
            << ", Eigen::Dynamic,Eigen::Dynamic>";
        is_template_type = true;
        break;
      case VOID_T:
        out << "void";
        break;
      default:
        out << "UNKNOWN TYPE";
      }

      for (size_t d = 0; d < t.num_dims_; ++d) {
        if (d > 0 || is_template_type)
          out << " ";
        out << ">";
      }
    }

    void generate_arg_decl(bool gen_const,
                           bool gen_ref,
                           const arg_decl& decl,
                           const std::string& scalar_t_name,
                           std::ostream& out) {
      if (gen_const)
        out << "const ";
      generate_bare_type(decl.arg_type_, scalar_t_name, out);
      if (gen_ref)
        out << "&";
      out << " " << decl.name_;
    }

    bool has_only_int_args(const function_decl_def& fun) {
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i)
        if (fun.arg_decls_[i].arg_type_.base_type_ != INT_T)
          return false;
      return true;
    }

    std::string fun_scalar_type(const function_decl_def& fun,
                                bool is_lp) {
      size_t num_args = fun.arg_decls_.size();
      // nullary, non-lp
      if (has_only_int_args(fun) && !is_lp)
        return "double";

      // need template metaprogram to construct return
      std::stringstream ss;
      ss << "typename boost::math::tools::promote_args<";
      int num_open_brackets = 1;
      int num_generated_params = 0;
      for (size_t i = 0; i < num_args; ++i) {
        if (fun.arg_decls_[i].arg_type_.base_type_ != INT_T) {
          // two conditionals cut and pasted below
          if (num_generated_params > 0)
            ss << ", ";
          if (num_generated_params == 4) {
            ss << "typename boost::math::tools::promote_args<";
            num_generated_params = 0;
            ++num_open_brackets;
          }
          ss << "T" << i << "__";
          ++num_generated_params;
        }
      }
      if (is_lp) {
        if (num_generated_params > 0)
          ss << ", ";
        // set threshold at 4 so always room for one more param at end
        ss << "T_lp__";
      }
      for (int i = 0; i < num_open_brackets; ++i)
        ss << ">::type";
      return ss.str();
    }
    // copy/modify for conditional_op???


    bool needs_template_params(const function_decl_def& fun) {
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i) {
        if (fun.arg_decls_[i].arg_type_.base_type_ != INT_T) {
          return true;
        }
      }
      return false;
    }


    void generate_function_template_parameters(const function_decl_def& fun,
                                               bool is_rng,
                                               bool is_lp,
                                               bool is_log,
                                               std::ostream& out) {
      if (needs_template_params(fun)) {
        out << "template <";
        bool continuing_tps = false;
        if (is_log) {
          out << "bool propto";
          continuing_tps = true;
        }
        for (size_t i = 0; i < fun.arg_decls_.size(); ++i) {
          // no template parameter for int-based args
          if (fun.arg_decls_[i].arg_type_.base_type_ != INT_T) {
            if (continuing_tps)
              out << ", ";
            out << "typename T" << i << "__";
            continuing_tps = true;
          }
        }
        if (is_rng) {
          if (continuing_tps)
            out << ", ";
          out << "class RNG";
          continuing_tps = true;
        } else if (is_lp) {
          if (continuing_tps)
            out << ", ";
          out << "typename T_lp__, typename T_lp_accum__";
          continuing_tps = true;
        }
        out << ">" << EOL;
      } else {  // no-arg function
        if (is_rng) {
          // nullary RNG case
          out << "template <class RNG>" << EOL;
        } else if (is_lp) {
          out << "template <typename T_lp__, typename T_lp_accum__>"
              << EOL;
        } else if (is_log) {
          out << "template <bool propto>"
              << EOL;
        }
      }
    }

    void generate_function_inline_return_type(const function_decl_def& fun,
                                              const std::string& scalar_t_name,
                                              int indent,
                                              std::ostream& out) {
      generate_indent(indent, out);
      generate_bare_type(fun.return_type_, scalar_t_name, out);
      out << EOL;
    }

    void generate_function_name(const function_decl_def& fun,
                                std::ostream& out) {
      out << fun.name_;
    }


    void generate_function_arguments(const function_decl_def& fun,
                                     bool is_rng,
                                     bool is_lp,
                                     bool is_log,
                                     std::ostream& out) {
      // arguments
      out << "(";
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i) {
        std::string template_type_i
          = "T" + boost::lexical_cast<std::string>(i) + "__";
        generate_arg_decl(true, true, fun.arg_decls_[i], template_type_i, out);
        if (i + 1 < fun.arg_decls_.size()) {
          out << "," << EOL << INDENT;
          for (size_t i = 0; i <= fun.name_.size(); ++i)
            out << " ";
        }
      }
      if ((is_rng || is_lp) && fun.arg_decls_.size() > 0)
        out << ", ";
      if (is_rng)
        out << "RNG& base_rng__";
      else if (is_lp)
        out << "T_lp__& lp__, T_lp_accum__& lp_accum__";
      if (is_rng || is_lp || fun.arg_decls_.size() > 0)
        out << ", ";
      out << "std::ostream* pstream__";
      out << ")";
    }

    void generate_functor_arguments(const function_decl_def& fun,
                                    bool is_rng,
                                    bool is_lp,
                                    bool is_log,
                                    std::ostream& out) {
      // arguments
      out << "(";
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i) {
        if (i > 0)
          out << ", ";
        out << fun.arg_decls_[i].name_;
      }
      if ((is_rng || is_lp) && fun.arg_decls_.size() > 0)
        out << ", ";
      if (is_rng)
        out << "base_rng__";
      else if (is_lp)
        out << "lp__, lp_accum__";
      if (is_rng || is_lp || fun.arg_decls_.size() > 0)
        out << ", ";
      out << "pstream__";
      out << ")";
    }



    void generate_function_body(const function_decl_def& fun,
                                const std::string& scalar_t_name,
                                std::ostream& out) {
      // no-op body
      if (fun.body_.is_no_op_statement()) {
        out << ";" << EOL;
        return;
      }
      out << " {" << EOL;
      out << INDENT
          << "typedef " << scalar_t_name << " fun_scalar_t__;"
          << EOL;
      out << INDENT
          << "typedef "
          << ((fun.return_type_.base_type_ == INT_T)
              ? "int" : "fun_scalar_t__")
          << " fun_return_scalar_t__;"
          << EOL;
      out << INDENT
          << "const static bool propto__ = true;"
          << EOL
          << INDENT
          << "(void) propto__;"
          << EOL;
      // use this dummy for inits
      out << INDENT2
          << "fun_scalar_t__ "
          << "DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      out << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;
      bool is_var_context = false;
      bool is_fun_return = true;
      bool include_sampling = true;
      out << INDENT
          << "int current_statement_begin__ = -1;"
          << EOL;

      generate_located_statement(fun.body_, 1, out,
                                 include_sampling, is_var_context,
                                 is_fun_return);

      out << "}"
          << EOL;
    }
    void generate_propto_default_function_body(const function_decl_def& fun,
                                               std::ostream& out) {
      out << " {" << EOL;
      out << INDENT << "return ";
      out << fun.name_ << "<false>(";
      for (size_t i = 0; i < fun.arg_decls_.size(); ++i) {
        if (i > 0)
          out << ",";
        out << fun.arg_decls_[i].name_;
      }
      if (fun.arg_decls_.size() > 0)
        out << ", ";
      out << "pstream__";
      out << ");" << EOL;
      out << "}" << EOL;
    }

    void generate_propto_default_function(const function_decl_def& fun,
                                          const std::string& scalar_t_name,
                                          std::ostream& out) {
      generate_function_template_parameters(fun, false, false, false, out);
      generate_function_inline_return_type(fun, scalar_t_name, 0, out);
      generate_function_name(fun, out);
      generate_function_arguments(fun, false, false, false, out);
      generate_propto_default_function_body(fun, out);
    }

    /**
     * Generate the specified function and optionally its default for
     * propto=false for functions ending in _log.
     *
     * Exact behavior differs for unmarked functions, and functions
     * ending in one of "_rng", "_lp", or "_log".
     *
     * @param[in] fun function AST object
     * @param[in, out] out output stream to which function definition
     * is written
     */
    void generate_function(const function_decl_def& fun,
                           std::ostream& out) {
      bool is_rng = ends_with("_rng", fun.name_);
      bool is_lp = ends_with("_lp", fun.name_);
      bool is_pf = ends_with("_log", fun.name_)
        || ends_with("_lpdf", fun.name_) || ends_with("_lpmf", fun.name_);
      std::string scalar_t_name = fun_scalar_type(fun, is_lp);

      generate_function_template_parameters(fun, is_rng, is_lp, is_pf, out);
      generate_function_inline_return_type(fun, scalar_t_name, 0, out);
      generate_function_name(fun, out);
      generate_function_arguments(fun, is_rng, is_lp, is_pf, out);
      generate_function_body(fun, scalar_t_name, out);

      // need a second function def for default propto=false for _log
      // funs; but don't want duplicate def, so don't do it for
      // forward decl when body is no-op
      if (is_pf && !fun.body_.is_no_op_statement())
        generate_propto_default_function(fun, scalar_t_name, out);
      out << EOL;
    }

    void generate_function_functor(const function_decl_def& fun,
                                   std::ostream& out) {
      if (fun.body_.is_no_op_statement())
        return;   // forward declaration, so no functor needed

      bool is_rng = ends_with("_rng", fun.name_);
      bool is_lp = ends_with("_lp", fun.name_);
      bool is_pf = ends_with("_log", fun.name_)
        || ends_with("_lpdf", fun.name_) || ends_with("_lpmf", fun.name_);
      std::string scalar_t_name = fun_scalar_type(fun, is_lp);

      out << EOL << "struct ";
      generate_function_name(fun, out);
      out << "_functor__ {" << EOL;

      out << INDENT;
      generate_function_template_parameters(fun, is_rng, is_lp, is_pf, out);

      out << INDENT;
      generate_function_inline_return_type(fun, scalar_t_name, 1, out);

      out <<  INDENT << "operator()";
      generate_function_arguments(fun, is_rng, is_lp, is_pf, out);
      out << " const {" << EOL;

      out << INDENT2 << "return ";
      generate_function_name(fun, out);
      generate_functor_arguments(fun, is_rng, is_lp, is_pf, out);
      out << ";" << EOL;
      out << INDENT << "}"  << EOL;
      out << "};" << EOL2;
    }


    void generate_functions(const std::vector<function_decl_def>& funs,
                            std::ostream& out) {
      for (size_t i = 0; i < funs.size(); ++i) {
        generate_function(funs[i], out);
        generate_function_functor(funs[i], out);
      }
    }

    void generate_member_var_decls_all(const program& prog,
                                       std::ostream& out) {
      generate_member_var_decls(prog.data_decl_, 1, out);
      generate_member_var_decls(prog.derived_data_decl_.first, 1, out);
    }

    void generate_globals(std::ostream& out) {
      out << "static int current_statement_begin__;"
          << EOL2;
    }


    void generate_cpp(const program& prog,
                      const std::string& model_name,
                      std::ostream& out) {
      generate_version_comment(out);
      generate_includes(out);
      generate_start_namespace(model_name, out);
      generate_usings(out);
      generate_typedefs(out);
      generate_globals(out);
      generate_functions(prog.function_decl_defs_, out);
      generate_class_decl(model_name, out);
      generate_private_decl(out);
      generate_member_var_decls_all(prog, out);
      generate_public_decl(out);
      generate_constructor(prog, model_name, out);
      generate_destructor(model_name, out);
      // put back if ever need integer params
      // generate_set_param_ranges(prog.parameter_decl_, out);
      generate_init_method(prog.parameter_decl_, out);
      generate_log_prob(prog, out);
      generate_param_names_method(prog, out);
      generate_dims_method(prog, out);
      generate_write_array_method(prog, model_name, out);
      generate_model_name_method(model_name, out);
      generate_constrained_param_names_method(prog, out);
      generate_unconstrained_param_names_method(prog, out);
      generate_end_class_decl(out);
      generate_end_namespace(out);
      generate_model_typedef(model_name, out);
    }

  }

}

#endif
