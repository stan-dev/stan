#ifndef __STAN__GM__PARSER_HPP__
#define __STAN__GM__PARSER_HPP__

#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/recursive_variant.hpp>

#include <iomanip>
#include <iostream>
#include <istream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>

// ADAPT must be in global namespace 

// not using adaptation relies on unary constructor

BOOST_FUSION_ADAPT_STRUCT(stan::gm::int_literal,
                          (int,val_)
                          (stan::gm::expr_type,type_))

BOOST_FUSION_ADAPT_STRUCT(stan::gm::double_literal,
                          (double,val_)
                          (stan::gm::expr_type,type_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::variable,
                          (std::string,name_)
                          (stan::gm::expr_type,type_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::int_var_decl,
                          (stan::gm::range, range_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::double_var_decl,
                          (stan::gm::range, range_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::vector_var_decl,
                          (stan::gm::expression, M_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::row_vector_var_decl,
                          (stan::gm::expression, N_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::matrix_var_decl,
                          (stan::gm::expression, M_)
                          (stan::gm::expression, N_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::simplex_var_decl,
                          (stan::gm::expression, K_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::pos_ordered_var_decl,
                          (stan::gm::expression, K_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::cov_matrix_var_decl,
                          (stan::gm::expression, K_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::corr_matrix_var_decl,
                          (stan::gm::expression, K_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::variable_dims,
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::fun,
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, args_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::index_op,
                          (stan::gm::expression, expr_)
                          (std::vector<std::vector<stan::gm::expression> >, dimss_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::range,
                          (stan::gm::expression, low_)
                          (stan::gm::expression, high_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::for_statement,
                          (std::string, variable_)
                          (stan::gm::range, range_)
                          (stan::gm::statement, statement_) )

namespace {
  // hack to pass pair into macro below to adapt
  struct DUMMY_STRUCT {
    typedef std::pair<std::vector<stan::gm::var_decl>,std::vector<stan::gm::statement> > type;
  };
}

BOOST_FUSION_ADAPT_STRUCT(stan::gm::program,
                          (std::vector<stan::gm::var_decl>, data_decl_)
                          (DUMMY_STRUCT::type, derived_data_decl_)
                          (std::vector<stan::gm::var_decl>, parameter_decl_)
                          (DUMMY_STRUCT::type, derived_decl_)
                          (stan::gm::statement, statement_)
                          (DUMMY_STRUCT::type, generated_decl_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::distribution,
                          (std::string, family_)
                          (std::vector<stan::gm::expression>, args_) )


BOOST_FUSION_ADAPT_STRUCT(stan::gm::statements,
                          (std::vector<stan::gm::var_decl>, local_decl_)
                          (std::vector<stan::gm::statement>, statements_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::sample,
                          (stan::gm::expression, expr_)
                          (stan::gm::distribution, dist_) 
                          (stan::gm::range, truncation_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::assignment,
                          (stan::gm::variable_dims, var_dims_)
                          (stan::gm::expression, expr_) )



namespace stan {
  
  namespace gm {

    // Cut-and-Paste from Spirit examples, including comment:  We
    // should be using expression::operator-. There's a bug in phoenix
    // type deduction mechanism that prevents us from doing
    // so. Phoenix will be switching to BOOST_TYPEOF. In the meantime,
    // we will use a phoenix::function below:
    struct negate_expr {
      template <typename T>
      struct result { typedef T type; };

      expression operator()(const expression& expr) const {
        return expression(unary_op('-', expr));
      }
    };
    boost::phoenix::function<negate_expr> neg;

    struct add_lp_var {
      template <typename T>
      struct result { typedef void type; };
      void operator()(variable_map& vm) const {
        vm.add("lp__",
               base_var_decl("lp__",std::vector<expression>(),DOUBLE_T));
      }
    };
    boost::phoenix::function<add_lp_var> add_lp_var_f;

    struct remove_lp_var {
      template <typename T>
      struct result { typedef void type; };
      void operator()(variable_map& vm) const {
        vm.remove("lp__");
      }
    };
    boost::phoenix::function<remove_lp_var> remove_lp_var_f;

    struct validate_no_constraints_vis : public boost::static_visitor<bool> {
      std::stringstream& error_msgs_;
      validate_no_constraints_vis(std::stringstream& error_msgs)
        : error_msgs_(error_msgs) { 
      }
      bool operator()(const nil& x) const { 
        error_msgs_ << "nil declarations not allowed";
        return false; // fail if arises
      } 
      bool operator()(const int_var_decl& x) const {
        if (x.range_.has_low() || x.range_.has_high()) {
          error_msgs_ << "require unconstrained."
                      << " found range constraint." << std::endl;
          return false;
        }
        return true;
      }
      bool operator()(const double_var_decl& x) const {
        if (x.range_.has_low() || x.range_.has_high()) {
          error_msgs_ << "require unconstrained."
                      << " found range constraint." << std::endl;
          return false;
        }
        return true;
      }
      bool operator()(const vector_var_decl& x) const {
        return true;
      }
      bool operator()(const row_vector_var_decl& x) const {
        return true;
      }
      bool operator()(const matrix_var_decl& x) const {
        return true;
      }
      bool operator()(const simplex_var_decl& x) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found simplex." << std::endl;
        return false;
      }
      bool operator()(const pos_ordered_var_decl& x) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found pos_ordered." << std::endl;
        return false;
      }
      bool operator()(const cov_matrix_var_decl& x) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found cov_matrix." << std::endl;
        return false;
      }
      bool operator()(const corr_matrix_var_decl& x) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found corr_matrix." << std::endl;
        return false;
      }
    };

    struct validate_decl_constraints {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef bool type; };

      bool operator()(const bool& allow_constraints,
                      const bool& declaration_ok,
                      const var_decl& var_decl,
                      std::stringstream& error_msgs) const {
        if (allow_constraints)
          return declaration_ok;
        if (!declaration_ok)
          return false; // short-circuits test of constraints
        validate_no_constraints_vis vis(error_msgs);
        bool constraints_ok = boost::apply_visitor(vis,var_decl.decl_);
        return constraints_ok;
      }
    };
    boost::phoenix::function<validate_decl_constraints> validate_decl_constraints_f;

    struct validate_allow_sample {
      template <typename T1, typename T2>
      struct result { typedef bool type; };

      bool operator()(const bool& allow_sample,
                      std::stringstream& error_msgs) const {
        if (!allow_sample) {
          error_msgs << "ERROR:  sampling only allowed in model."
                     << std::endl;
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_allow_sample> validate_allow_sample_f;

    struct validate_int_expr {
      template <typename T1, typename T2>
      struct result { typedef bool type; };

      bool operator()(const expression& expr,
                      std::stringstream& error_msgs) const {
        if (!expr.expression_type().is_primitive_int()) {
          error_msgs << "expression denoting integer required; found type=" 
                     << expr.expression_type() << std::endl;
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_int_expr> validate_int_expr_f;

    struct validate_double_expr {
      template <typename T1, typename T2>
      struct result { typedef bool type; };

      bool operator()(const expression& expr,
                      std::stringstream& error_msgs) const {
        if (!expr.expression_type().is_primitive_double()
            && !expr.expression_type().is_primitive_int()) {
          error_msgs << "expression denoting double required; found type=" 
                     << expr.expression_type() << std::endl;
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_double_expr> validate_double_expr_f;

    struct validate_expr_type {
      template <typename T>
      struct result { typedef bool type; };

      bool operator()(const expression& expr) const {
        return !expr.expression_type().is_ill_formed();
      }
    };
    boost::phoenix::function<validate_expr_type> validate_expr_type_f;

    

    struct validate_assignment {
      template <typename T1, typename T2, typename T3>
      struct result { typedef bool type; };

      bool operator()(assignment& a,
                      variable_map& vm,
                      std::stringstream& error_msgs) const {

        std::string name = a.var_dims_.name_;
        if (!vm.exists(name)) {
          error_msgs << "unknown variable in assignment"
                     << "; lhs variable=" << a.var_dims_.name_ 
                     << std::endl;
          return false;
        }
        a.var_type_ = vm.get(name);
        unsigned int lhs_var_num_dims = a.var_type_.dims_.size();
        unsigned int num_index_dims = a.var_dims_.dims_.size();

        expr_type lhs_type = infer_type_indexing(a.var_type_.base_type_,
                                                 lhs_var_num_dims,
                                                 num_index_dims);

        if (lhs_type.is_ill_formed()
            || lhs_type.num_dims_ != a.expr_.expression_type().num_dims_) {
          error_msgs << "too many indices on left-hand-side of assignment"
                     << "; lhs variable=" << a.var_dims_.name_ 
                     << "; base type=" << a.var_type_.base_type_
                     << "; num dims=" << lhs_var_num_dims
                     << "; num indices=" << num_index_dims
                     << std::endl;
          return false;
        }

        base_expr_type lhs_base_type = lhs_type.base_type_;
        base_expr_type rhs_base_type = a.expr_.expression_type().base_type_;
        bool types_compatible 
          = lhs_base_type == rhs_base_type
          || ( lhs_base_type == DOUBLE_T && rhs_base_type == INT_T ); // int -> double promotion
        if (!types_compatible) {
          error_msgs << "base type mismatch in assignment"
                     << "; lhs variable=" << a.var_dims_.name_
                     << "; lhs base type=" << lhs_base_type
                     << "; rhs base type=" << rhs_base_type
                     << std::endl;
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_assignment> validate_assignment_f;

    struct validate_sample {
      template <typename T>
      struct result { typedef bool type; };

      bool operator()(const sample& s) const {
        std::vector<expr_type> arg_types;
        arg_types.push_back(s.expr_.expression_type());
        for (unsigned int i = 0; i < s.dist_.args_.size(); ++i)
          arg_types.push_back(s.dist_.args_[i].expression_type());
        std::string function_name(s.dist_.family_);
        function_name += "_log";
        expr_type result_type 
          = function_signatures::instance().get_result_type(function_name,arg_types);
        return result_type.is_primitive_double();
      }
    };
    boost::phoenix::function<validate_sample> validate_sample_f;

    struct validate_primitive_int_type {
      template <typename T>
      struct result { typedef bool type; };

      bool operator()(const expression& expr) const {
        return expr.expression_type().is_primitive_int();
      }
    };
    boost::phoenix::function<validate_primitive_int_type> validate_primitive_int_type_f;

    struct set_var_type {
      template <typename T1, typename T2>
      struct result { typedef variable type; };
      variable operator()(variable& var_expr, 
                          variable_map& vm) const {
        std::string name = var_expr.name_;
        if (!vm.exists(name)) {
          throw std::runtime_error("variable does not exist");  
          // FIXME: patch with pass = false and thread
        }
        var_expr.set_type(vm.get_base_type(name),vm.get_num_dims(name));
        return var_expr;
      }
    };
    boost::phoenix::function<set_var_type> set_var_type_f;

    struct unscope_locals {
      template <typename T1, typename T2>
      struct result { typedef void type; };
      void operator()(const std::vector<var_decl>& var_decls,
                      variable_map& vm) const {
        for (unsigned int i = 0; i < var_decls.size(); ++i)
          vm.remove(var_decls[i].name());
      }
    };
    boost::phoenix::function<unscope_locals> unscope_locals_f;


    struct set_fun_type {
      template <typename T1>
      struct result { typedef T1 type; };

      fun operator()(fun& fun) const {
        std::vector<expr_type> arg_types;
        for (unsigned int i = 0; i < fun.args_.size(); ++i)
          arg_types.push_back(fun.args_[i].expression_type());
        fun.type_ = function_signatures::instance().get_result_type(fun.name_,
                                                                    arg_types);
        return fun;
      }
    };
    boost::phoenix::function<set_fun_type> set_fun_type_f;


    struct add_var {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef T1 type; };
      template <typename T>
      T operator()(const T& var_decl, 
                   variable_map& vm,
                   bool& pass,
                   const var_origin& vo) const {
        if (vm.exists(var_decl.name_)) {
          // variable already exists
          pass = false;
          return var_decl;
        }
        pass = true;  // probably don't need to set true
        vm.add(var_decl.name_,var_decl);
        return var_decl;
      }
    };
    boost::phoenix::function<add_var> add_var_f;

    struct add_loop_identifier {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef bool type; };
      bool operator()(const std::string& name, 
                      std::string& name_local,
                      variable_map& vm,
                      std::stringstream& error_msgs) const {
        name_local = name;
        if (vm.exists(name)) {
          error_msgs << "ERROR: loop variable already declared."
                     << " variable name=\"" << name << "\"" << std::endl;
          return false; // variable exists
        }
        vm.add(name, base_var_decl(name,std::vector<expression>(),INT_T));
        return true;
      }
    };
    boost::phoenix::function<add_loop_identifier> add_loop_identifier_f;

    struct remove_loop_identifier {
      template <typename T1, typename T2>
      struct result { typedef void type; };
      void operator()(const std::string& name, 
                      variable_map& vm) const {
        vm.remove(name);
      }
    };
    boost::phoenix::function<remove_loop_identifier> remove_loop_identifier_f;

    struct set_indexed_factor_type {
      template <typename T>
      struct result { typedef bool type; };
      bool operator()(index_op& io) const {
        io.infer_type();
        return !io.type_.is_ill_formed();
      }
    };
    boost::phoenix::function<set_indexed_factor_type> set_indexed_factor_type_f;

    template <typename Iterator>
    class whitespace_grammar : public qi::grammar<Iterator> {
    public:
      whitespace_grammar() : whitespace_grammar::base_type(whitespace) {
        whitespace 
          = ( qi::omit["/*"] >> *(qi::char_ - "*/") > qi::omit["*/"] )
          | ( qi::omit["//"] >> *(qi::char_ - qi::eol) )
          | ( qi::omit["#"] >> *(qi::char_ - qi::eol) )
          | ascii::space_type()
          ;
      }
    private:
      qi::rule<Iterator> whitespace;
    };


    template <typename Iterator>
    struct program_grammar : qi::grammar<Iterator, 
                                         program(), 
                                         whitespace_grammar<Iterator> > {
 
      program_grammar() 
        : program_grammar::base_type(program_r) {
        using qi::_val;
        using qi::_1;
        using qi::_pass;
        using qi::double_;
        using qi::int_;
        using boost::spirit::qi::eps;
        using namespace qi::labels;

        program_r.name("program");
        program_r 
          %= -data_var_decls_r
          > -derived_data_var_decls_r
          > -param_var_decls_r
          // scope lp__ to "transformed params" and "model" only
          > qi::eps[add_lp_var_f(boost::phoenix::ref(var_map_))]
          > -derived_var_decls_r
          > model_r
          > qi::eps[remove_lp_var_f(boost::phoenix::ref(var_map_))]
          > -generated_var_decls_r
          ;
        
        model_r.name("model declaration");
        model_r 
          %= qi::lit("model")
          > statement_r(true)
          ;

        data_var_decls_r.name("data variable declarations");
        data_var_decls_r
          %= qi::lit("data")
          > qi::lit('{')
          > *var_decl_r(true,data_origin) // +constraints
          > qi::lit('}');

        derived_data_var_decls_r.name("derived data variable declaration and statement");
        derived_data_var_decls_r
          %= qi::lit("derived")
          >> qi::lit("data")
          > qi::lit('{')
          > *var_decl_r(false,transformed_data_origin)  // -constraints
          > *statement_r(false) // -sampling
          > qi::lit('}');

        param_var_decls_r.name("parameter variable declarations");
        param_var_decls_r
          %= qi::lit("parameters")
          > qi::lit('{')
          > *var_decl_r(true,parameter_origin) // +constraints
          > qi::lit('}');

        derived_var_decls_r.name("derived variable declarations");
        derived_var_decls_r
          %= qi::lit("derived")
          >> qi::lit("parameters")
          > qi::lit('{')
          > *var_decl_r(false,transformed_parameter_origin) // -constraints
          > *statement_r(false) // -sampling
          > qi::lit('}');

        generated_var_decls_r.name("generated variable declarations");
        generated_var_decls_r
          %= qi::lit("generated")
          > qi::lit("quantities")
          > qi::lit('{')
          > *var_decl_r(false,derived_origin) // -constraints
          > *statement_r(false) // -sampling
          > qi::lit('}');

        // _a local to hold error state, _r1 inherited true if constriaints allowed
        var_decl_r.name("variable declaration");
        var_decl_r 
          %= (int_decl_r             [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | double_decl_r        [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | vector_decl_r        [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | row_vector_decl_r    [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | matrix_decl_r        [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | simplex_decl_r       [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | pos_ordered_decl_r   [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | corr_matrix_decl_r   [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | cov_matrix_decl_r    [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              )
          > qi::eps[_pass = validate_decl_constraints_f(_r1,_a,_val,boost::phoenix::ref(error_msgs_))]
          ;

        int_decl_r.name("integer declaration");
        int_decl_r 
          %= qi::lit("int")
          > -range_brackets_int_r
          > identifier_r 
          > opt_dims_r
          > qi::lit(';');

        double_decl_r.name("double declaration");
        double_decl_r 
          %= qi::lit("double")
          > -range_brackets_double_r
          > identifier_r
          > opt_dims_r
          > qi::lit(';');

        vector_decl_r.name("vector declaration");
        vector_decl_r 
          %= qi::lit("vector")
          > qi::lit('(')
          > expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > qi::lit(')')
          > identifier_r 
          > opt_dims_r
          > qi::lit(';');

        row_vector_decl_r.name("row vector declaration");
        row_vector_decl_r 
          %= qi::lit("row_vector")
          > qi::lit('(')
          > expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > qi::lit(')')
          > identifier_r 
          > opt_dims_r
          > qi::lit(';');

        matrix_decl_r.name("matrix declaration");
        matrix_decl_r 
          %= qi::lit("matrix")
          > qi::lit('(')
          > expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > qi::lit(',')
          > expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > qi::lit(')')
          > identifier_r 
          > opt_dims_r
          > qi::lit(';');

        simplex_decl_r.name("simplex declaration");
        simplex_decl_r 
          %= qi::lit("simplex")
          > qi::lit('(')
          > expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > qi::lit(')')
          > identifier_r 
          > opt_dims_r
          > qi::lit(';');

        pos_ordered_decl_r.name("positive ordered declaration");
        pos_ordered_decl_r 
          %= qi::lit("pos_ordered")
          > qi::lit('(')
          > expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > qi::lit(')')
          > identifier_r 
          > opt_dims_r
          > qi::lit(';');

        cov_matrix_decl_r.name("positive definite symmetric (covariance) matrix declaration");
        cov_matrix_decl_r 
          %= qi::lit("cov_matrix")
          > qi::lit('(')
          > expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > qi::lit(')')
          > identifier_r 
          > opt_dims_r
          > qi::lit(';');

        corr_matrix_decl_r.name("correlation matrix declaration");
        corr_matrix_decl_r 
          %= qi::lit("corr_matrix")
          > qi::lit('(')
          > expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > qi::lit(')')
          > identifier_r 
          > opt_dims_r
          > qi::lit(';');

        expression_r.name("expression");
        expression_r 
          %=  term_r                          [_val = _1]
          >> *( (qi::lit('+') > term_r        [_val += _1])
                |   (qi::lit('-') > term_r    [_val -= _1])
                )
          > qi::eps[_pass = validate_expr_type_f(_val)];
          ;

        term_r.name("term");
        term_r 
          %= ( negated_factor_r                          [_val = _1]
              >> *( (qi::lit('*') > negated_factor_r     [_val *= _1])
                    | (qi::lit('/') > negated_factor_r   [_val /= _1])
                    )
              )
          ;

        negated_factor_r 
          %= qi::lit('-') >> indexed_factor_r [_val = neg(_1)]
          | qi::lit('+') >> indexed_factor_r [_val = _1]
          | indexed_factor_r [_val = _1];
        
        // two of these to put semantic action on this one w. index_op input
        indexed_factor_r.name("(optionally) indexed factor [sub]");
        indexed_factor_r 
          %= indexed_factor_2_r [_pass = set_indexed_factor_type_f(_1)];

        indexed_factor_2_r.name("(optionally) indexed factor [sub] 2");
        indexed_factor_2_r 
          %= (factor_r >> *dims_r);

        factor_r.name("factor");
        factor_r
          %= int_literal_r      [_val = _1]
          | double_literal_r    [_val = _1]
          | fun_r               [_val = set_fun_type_f(_1)]
          | variable_r          [_val = set_var_type_f(_1,boost::phoenix::ref(var_map_))]
          | ( qi::lit('(') 
              > expression_r    [_val = _1] 
              > qi::lit(')') )
          ;

        int_literal_r.name("integer literal");
        int_literal_r
          %= int_ 
             >> !( qi::lit('.')
                   | qi::lit('e')
                   | qi::lit('E') );

        double_literal_r.name("double literal");
        double_literal_r
          %= double_;


        // no optional dims in the variable_r
        variable_r.name("variable expression");
        variable_r
          %= identifier_r;

        fun_r.name("function and argument expressions");
        fun_r 
          %= identifier_r 
          >> args_r; 
            
        opt_dims_r.name("array dimensions (optional)");
        opt_dims_r 
          %=  - dims_r;

        dims_r.name("array dimensions");
        dims_r 
          %= qi::lit('[') 
          > (expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
             % ',')
          > qi::lit(']')
          ;
        
        range_r.name("range expression pair, colon");
        range_r 
          %= expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          >> qi::lit(':') 
          >> expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))];

        truncation_range_r.name("range pair");
        truncation_range_r
          %= qi::lit('T')
          > qi::lit('(') 
          > -expression_r
          > qi::lit(',')
          > -expression_r
          > qi::lit(')');
        
        range_brackets_int_r.name("range expression pair, brackets");
        range_brackets_int_r 
          %= qi::lit('(') 
          > -(expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))])
          > qi::lit(',')
          > -(expression_r [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))])
          > qi::lit(')');

        range_brackets_double_r.name("range expression pair, brackets");
        range_brackets_double_r 
          %= qi::lit('(') 
          > -(expression_r [_pass = validate_double_expr_f(_1,boost::phoenix::ref(error_msgs_))])
          > qi::lit(',')
          > -(expression_r [_pass = validate_double_expr_f(_1,boost::phoenix::ref(error_msgs_))])
          > qi::lit(')');

        args_r.name("function argument expressions");
        args_r 
          %= qi::lit('(') 
          >> (expression_r % ',')
          > qi::lit(')');

        identifier_r.name("identifier");
        identifier_r
          %= (qi::lexeme[qi::char_("a-zA-Z") 
                        >> *qi::char_("a-zA-Z0-9_.")]);

        distribution_r.name("distribution and parameters");
        distribution_r
          %= identifier_r
          >> qi::lit('(')
          >> -(expression_r % ',')
          > qi::lit(')');

        // inherited  _r1 = true if samples allowed as statements
        sample_r.name("distribution of expression");
        sample_r 
          %= expression_r
          >> qi::lit('~')
          > qi::eps[_pass = validate_allow_sample_f(_r1,boost::phoenix::ref(error_msgs_))] 
          > distribution_r
          > -truncation_range_r
          > qi::lit(';');
        
        var_lhs_r.name("variable and array dimensions");
        var_lhs_r 
          %= identifier_r 
          >> opt_dims_r;

        assignment_r.name("variable assignment by expression");
        assignment_r
          %= var_lhs_r
          >> qi::lit("<-")
          > expression_r
          > qi::lit(';') 
          ;

        // inherited attribute _r1 is type bool
        // set to true if sample_r are allowed
        statement_r.name("statement");
        statement_r
          %= statement_seq_r(_r1)
          | for_statement_r(_r1)
          | assignment_r [_pass 
                          = validate_assignment_f(_1,boost::phoenix::ref(var_map_),
                                                  boost::phoenix::ref(error_msgs_))]
          | sample_r(_r1) [_pass = validate_sample_f(_1)]
          | no_op_statement_r
          ;

        no_op_statement_r.name("no op statement");
        no_op_statement_r 
          %= qi::lit(';') [_val = no_op_statement()];  // ok to re-use instance

        for_statement_r.name("for statement");
        for_statement_r
          %= qi::lit("for")
          > qi::lit('(')
          > identifier_r [_pass 
                          = add_loop_identifier_f(_1,_a,
                                                  boost::phoenix::ref(var_map_),
                                                  boost::phoenix::ref(error_msgs_))]
          > qi::lit("in")
          > range_r
          > qi::lit(')')
          > statement_r(_r1)
          > qi::eps [remove_loop_identifier_f(_a,boost::phoenix::ref(var_map_))];
          ;

          // _r1 = true if sampling statements allowed
        statement_seq_r.name("sequence of statements");
        statement_seq_r
          %= qi::lit('{')
          > local_var_decls_r[_a = _1]
          > *statement_r(_r1)
          > qi::lit('}')
          > qi::eps[unscope_locals_f(_a,boost::phoenix::ref(var_map_))]
          ;

        local_var_decls_r
          %= *var_decl_r(false,local_origin); // - constants

        // hack cast to write to error_msgs_ of type stringstream
        qi::on_error<qi::rethrow>(var_decl_r,
                                  (std::ostream&)error_msgs_
                                  << boost::phoenix::val("ERROR: Ill-formed variable declaration.")
                                  << std::endl);

        qi::on_error<qi::rethrow>(indexed_factor_r,
                                  (std::ostream&)error_msgs_
                                  << boost::phoenix::val("ERROR: Ill-formed factor.")
                                  << std::endl);

        qi::on_error<qi::rethrow>(program_r,
                                  (std::ostream&)error_msgs_
                                  << boost::phoenix::val("ERROR: Expected ")
                                  << _4 
                                  << std::endl);
      }

      // global info for parses
      variable_map var_map_;
      std::stringstream error_msgs_;

      // rules
      qi::rule<Iterator, expression(), whitespace_grammar<Iterator> > expression_r;
      qi::rule<Iterator, expression(), whitespace_grammar<Iterator> > term_r;
      qi::rule<Iterator, expression(), whitespace_grammar<Iterator> > factor_r;
      qi::rule<Iterator, variable(), whitespace_grammar<Iterator> > variable_r;
      qi::rule<Iterator, int_literal(), whitespace_grammar<Iterator> > int_literal_r;
      qi::rule<Iterator, double_literal(), whitespace_grammar<Iterator> > double_literal_r;
      qi::rule<Iterator, variable_dims(), whitespace_grammar<Iterator> > var_lhs_r;
      qi::rule<Iterator, fun(), whitespace_grammar<Iterator> > fun_r;
      qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > identifier_r;
      qi::rule<Iterator, std::vector<expression>(), whitespace_grammar<Iterator> > opt_dims_r;
      qi::rule<Iterator, std::vector<expression>(), whitespace_grammar<Iterator> > dims_r;
      qi::rule<Iterator, range(), whitespace_grammar<Iterator> > range_r;
      qi::rule<Iterator, range(), whitespace_grammar<Iterator> > truncation_range_r;
      qi::rule<Iterator, range(), whitespace_grammar<Iterator> > range_brackets_int_r;
      qi::rule<Iterator, range(), whitespace_grammar<Iterator> > range_brackets_double_r;
      qi::rule<Iterator, std::vector<expression>(), whitespace_grammar<Iterator> > args_r;
      qi::rule<Iterator, int_var_decl(), whitespace_grammar<Iterator> > int_decl_r;
      qi::rule<Iterator, double_var_decl(), whitespace_grammar<Iterator> > double_decl_r;
      qi::rule<Iterator, vector_var_decl(), whitespace_grammar<Iterator> > vector_decl_r;
      qi::rule<Iterator, row_vector_var_decl(), whitespace_grammar<Iterator> > row_vector_decl_r;
      qi::rule<Iterator, matrix_var_decl(), whitespace_grammar<Iterator> > matrix_decl_r;
      qi::rule<Iterator, simplex_var_decl(), whitespace_grammar<Iterator> > simplex_decl_r;
      qi::rule<Iterator, pos_ordered_var_decl(), whitespace_grammar<Iterator> > pos_ordered_decl_r;
      qi::rule<Iterator, cov_matrix_var_decl(), whitespace_grammar<Iterator> > cov_matrix_decl_r;
      qi::rule<Iterator, corr_matrix_var_decl(), whitespace_grammar<Iterator> > corr_matrix_decl_r;
      qi::rule<Iterator, qi::locals<bool>, var_decl(bool,var_origin), whitespace_grammar<Iterator> > var_decl_r;
      qi::rule<Iterator, std::vector<var_decl>(), whitespace_grammar<Iterator> > data_var_decls_r;
      qi::rule<Iterator, std::pair<std::vector<var_decl>,std::vector<statement> >(), 
               whitespace_grammar<Iterator> > derived_data_var_decls_r;
      qi::rule<Iterator, std::vector<var_decl>(), whitespace_grammar<Iterator> > param_var_decls_r;
      qi::rule<Iterator, std::pair<std::vector<var_decl>,std::vector<statement> >(), 
               whitespace_grammar<Iterator> > derived_var_decls_r;
      qi::rule<Iterator, std::pair<std::vector<var_decl>,std::vector<statement> >(), 
               whitespace_grammar<Iterator> > generated_var_decls_r;
      qi::rule<Iterator, std::vector<var_decl>(), whitespace_grammar<Iterator> > local_var_decls_r;
      qi::rule<Iterator, program(), whitespace_grammar<Iterator> > program_r;
      qi::rule<Iterator, distribution(), whitespace_grammar<Iterator> > distribution_r;
      qi::rule<Iterator, sample(bool), whitespace_grammar<Iterator> > sample_r;
      qi::rule<Iterator, assignment(), whitespace_grammar<Iterator> > assignment_r;
      qi::rule<Iterator, statement(bool), whitespace_grammar<Iterator> > statement_r;
      qi::rule<Iterator, qi::locals<std::vector<var_decl> >, 
               statements(bool), whitespace_grammar<Iterator> > statement_seq_r;
      qi::rule<Iterator, qi::locals<std::string>, for_statement(bool), 
               whitespace_grammar<Iterator> > for_statement_r;
      qi::rule<Iterator, statement(), whitespace_grammar<Iterator> > model_r;
      qi::rule<Iterator, no_op_statement(), whitespace_grammar<Iterator> > no_op_statement_r;
      qi::rule<Iterator, expression(), whitespace_grammar<Iterator> > indexed_factor_r;
      // two of these because of type-coercion from index_op to expression
      qi::rule<Iterator, index_op(), whitespace_grammar<Iterator> > indexed_factor_2_r; 
      qi::rule<Iterator, expression(), whitespace_grammar<Iterator> > negated_factor_r;
    };

    // Cut and paste source for iterator & reporting pattern:
    // http://boost-spirit.com/home/articles/qi-example/tracking-the-input-position-while-parsing/
    // http://boost-spirit.com/dl_more/parsing_tracking_position/stream_iterator_errorposition_parsing.cpp
    bool parse(std::istream& input, 
               const std::string& filename, 
               program& result) {

      namespace classic = boost::spirit::classic;

      // iterate over stream input
      typedef std::istreambuf_iterator<char> base_iterator_type;
      base_iterator_type in_begin(input);
      
      // convert input iterator to forward iterator, usable by spirit parser
      typedef boost::spirit::multi_pass<base_iterator_type> forward_iterator_type;
      forward_iterator_type fwd_begin = boost::spirit::make_default_multi_pass(in_begin);
      forward_iterator_type fwd_end;
      
      // wrap forward iterator with position iterator, to record the position
      typedef classic::position_iterator2<forward_iterator_type> pos_iterator_type;
      pos_iterator_type position_begin(fwd_begin, fwd_end, filename);
      pos_iterator_type position_end;
      
      program_grammar<pos_iterator_type> prog_grammar;
      whitespace_grammar<pos_iterator_type> whitesp_grammar;
      
      bool success = 0;
      try {
        success = qi::phrase_parse(position_begin, 
                                   position_end,
                                   prog_grammar,
                                   whitesp_grammar,
                                   result);
      } catch (const qi::expectation_failure<pos_iterator_type>& e) {
        // FIXME: generalize beyond expectation failures
        const classic::file_position_base<std::string>& pos = e.first.get_position();
        std::stringstream msg;
        msg << "parse error at file " 
            << pos.file 
            << " line " 
            << pos.line 
            << " column " 
            << pos.column 
            << std::endl 
            << e.first.get_currentline() 
            << std::endl;
        for (int i = 2; i < pos.column; ++i)
          msg << ' ';
        msg << " ^-- here";
        msg << std::endl;
        msg << prog_grammar.error_msgs_;
        msg << std::endl;
        throw std::invalid_argument(msg.str());
      }
      return success && (position_begin == position_end); // want to consume ALL input
    }

  }

}

#endif
