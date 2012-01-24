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

#include <stan/gm/ast.hpp>

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
                          (std::vector<std::vector<stan::gm::expression> >, 
                           dimss_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::range,
                          (stan::gm::expression, low_)
                          (stan::gm::expression, high_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::for_statement,
                          (std::string, variable_)
                          (stan::gm::range, range_)
                          (stan::gm::statement, statement_) )

namespace {
  // hack to pass pair into macro below to adapt; in namespace to hide
  struct DUMMY_STRUCT {
    typedef std::pair<std::vector<stan::gm::var_decl>,
                      std::vector<stan::gm::statement> > type;
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
               base_var_decl("lp__",std::vector<expression>(),DOUBLE_T),
               local_origin); // lp acts as a local where defined
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
    boost::phoenix::function<validate_decl_constraints> 
    validate_decl_constraints_f;

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

    struct data_only_expression : public boost::static_visitor<bool> {
      std::stringstream& error_msgs_;
      variable_map& var_map_;
      data_only_expression(std::stringstream& error_msgs,
                           variable_map& var_map) 
        : error_msgs_(error_msgs),
          var_map_(var_map) {
      }
      bool operator()(const nil& e) const {
        return true;
      }
      bool operator()(const int_literal& x) const {
        return true;
      }
      bool operator()(const double_literal& x) const {
        return true;
      }
      bool operator()(const variable& x) const {
        var_origin origin = var_map_.get_origin(x.name_);
        bool is_data = (origin == data_origin) || (origin == transformed_data_origin);
        if (!is_data) {
          error_msgs_ << "non-data variables not allowed in dimension declarations."
                      << std::endl
                      << "     found variable=" << x.name_
                      << "; declared in block=" << origin
                      << std::endl;
        }
        return is_data;
      }
      bool operator()(const fun& x) const {
        for (unsigned int i = 0; i < x.args_.size(); ++i)
          if (!boost::apply_visitor(*this,x.args_[i].expr_))
            return false;
        return true;
      }
      bool operator()(const index_op& x) const {
        if (!boost::apply_visitor(*this,x.expr_.expr_))
          return false;
        for (unsigned int i = 0; i < x.dimss_.size(); ++i)
          for (unsigned int j = 0; j < x.dimss_[i].size(); ++j)
            if (!boost::apply_visitor(*this,x.dimss_[i][j].expr_))
              return false;
        return true;
      }
      bool operator()(const binary_op& x) const {
        return boost::apply_visitor(*this,x.left.expr_)
          && boost::apply_visitor(*this,x.right.expr_);
      }
      bool operator()(const unary_op& x) const {
        return boost::apply_visitor(*this,x.subject.expr_);
      }
    };

    // #include <stan/gm/generator.hpp>

    struct validate_int_data_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef bool type; };

      bool operator()(const expression& expr,
                      variable_map& var_map,
                      std::stringstream& error_msgs) const {
        if (!expr.expression_type().is_primitive_int()) {
          error_msgs << "dimension declaration requires expression denoting integer;"
                     << " found type=" 
                     << expr.expression_type() 
                     << std::endl;
          return false;
        }
        data_only_expression vis(error_msgs,var_map);
        bool only_data_dimensions = boost::apply_visitor(vis,expr.expr_);
        return only_data_dimensions;
      }
    };
    boost::phoenix::function<validate_int_data_expr> validate_int_data_expr_f;

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
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef bool type; };

      bool operator()(assignment& a,
                      const var_origin& origin_allowed,
                      variable_map& vm,
                      std::stringstream& error_msgs) const {

        // validate existence
        std::string name = a.var_dims_.name_;
        if (!vm.exists(name)) {
          error_msgs << "unknown variable in assignment"
                     << "; lhs variable=" << a.var_dims_.name_ 
                     << std::endl;
          return false;
        }
        
        // validate origin
        var_origin lhs_origin = vm.get_origin(name);
        if (lhs_origin != local_origin
            && lhs_origin != origin_allowed) {
          error_msgs << "attempt to assign variable in wrong block."
                     << " left-hand-side variable origin=" << lhs_origin
                     << std::endl;
          return false;
        }

        // validate types
        a.var_type_ = vm.get(name);
        unsigned int lhs_var_num_dims = a.var_type_.dims_.size();
        unsigned int num_index_dims = a.var_dims_.dims_.size();

        expr_type lhs_type = infer_type_indexing(a.var_type_.base_type_,
                                                 lhs_var_num_dims,
                                                 num_index_dims);

        if (lhs_type.is_ill_formed()) {
          error_msgs << "ill-formed lhs of assignment"
                     << std::endl;
          return false;
        }
        if (lhs_type.num_dims_ != a.expr_.expression_type().num_dims_) {
          error_msgs << "mismatched dimensions on left- and right-hand side of assignment"
                     << "; left dims=" << lhs_type.num_dims_
                     << "; right dims=" << a.expr_.expression_type().num_dims_
                     << std::endl;
          return false;
        }

        base_expr_type lhs_base_type = lhs_type.base_type_;
        base_expr_type rhs_base_type = a.expr_.expression_type().base_type_;
        // int -> double promotion
        bool types_compatible 
          = lhs_base_type == rhs_base_type
          || ( lhs_base_type == DOUBLE_T && rhs_base_type == INT_T );
        if (!types_compatible) {
          error_msgs << "base type mismatch in assignment"
                     << "; left variable=" << a.var_dims_.name_
                     << "; left base type=" << lhs_base_type
                     << "; right base type=" << rhs_base_type
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
          = function_signatures::instance()
          .get_result_type(function_name,arg_types);
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
    boost::phoenix::function<validate_primitive_int_type> 
    validate_primitive_int_type_f;

    struct set_var_type {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef variable type; };
      variable operator()(variable& var_expr, 
                          variable_map& vm,
                          std::ostream& error_msgs,
                          bool& pass) const {
        std::string name = var_expr.name_;
        if (!vm.exists(name)) {
          pass = false;
          error_msgs << "variable \"" << name << '"' << " does not exist." 
                     << std::endl;
          return var_expr;
        }
        pass = true;
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
      // each type derived from base_var_decl gets own instance
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
        vm.add(var_decl.name_,var_decl,vo);
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
        vm.add(name, 
               base_var_decl(name,std::vector<expression>(),
                             INT_T),
               local_origin); // loop var acts like local
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
    class whitespace_grammar : public boost::spirit::qi::grammar<Iterator> {
    public:
      whitespace_grammar() : whitespace_grammar::base_type(whitespace) {
        using boost::spirit::qi::omit;
        using boost::spirit::qi::char_;
        using boost::spirit::qi::eol;
        whitespace 
          = ( omit["/*"] >> *(char_ - "*/") > omit["*/"] )
          | ( omit["//"] >> *(char_ - eol) )
          | ( omit["#"] >> *(char_ - eol) )
          | boost::spirit::ascii::space_type()
          ;
      }
    private:
      boost::spirit::qi::rule<Iterator> whitespace;
    };


                              
                              

    template <typename Iterator>
    struct expression_grammar 
      : boost::spirit::qi::grammar<Iterator,
                                   expression(),
                                   whitespace_grammar<Iterator> > {

      expression_grammar(variable_map& var_map,
                         std::stringstream& error_msgs) 
      : expression_grammar::base_type(expression_r),
        var_map_(var_map),
        error_msgs_(error_msgs) {

        using boost::spirit::qi::_1;
        using boost::spirit::qi::char_;
        using boost::spirit::qi::double_;
        using boost::spirit::qi::eps;
        using boost::spirit::qi::int_;
        using boost::spirit::qi::lexeme;
        using boost::spirit::qi::lit;
        using boost::spirit::qi::_pass;
        using boost::spirit::qi::_val;

        expression_r.name("expression");
        expression_r 
          %=  term_r                          [_val = _1]
          >> *( (lit('+') > term_r        [_val += _1])
                |   (lit('-') > term_r    [_val -= _1])
                )
          > eps[_pass = validate_expr_type_f(_val)];
        ;

        term_r.name("term");
        term_r 
          %= ( negated_factor_r                          [_val = _1]
               >> *( (lit('*') > negated_factor_r     [_val *= _1])
                      | (lit('/') > negated_factor_r   [_val /= _1])
                     )
               )
          ;

        negated_factor_r 
          %= lit('-') >> indexed_factor_r [_val = neg(_1)]
          | lit('+') >> indexed_factor_r [_val = _1]
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
          %=  int_literal_r      [_val = _1]
          | double_literal_r    [_val = _1]
          | fun_r               [_val = set_fun_type_f(_1)]
          | variable_r          
            [_val = set_var_type_f(_1,boost::phoenix::ref(var_map_),
                                   boost::phoenix::ref(error_msgs_),
                                   _pass)]
          | ( lit('(') 
              > expression_r    [_val = _1]
              > lit(')') )
          ;

        int_literal_r.name("integer literal");
        int_literal_r
          %= int_ 
             >> !( lit('.')
                   | lit('e')
                   | lit('E') );

        double_literal_r.name("double literal");
        double_literal_r
          %= double_;

        fun_r.name("function and argument expressions");
        fun_r 
          %= identifier_r 
          >> args_r; 

        identifier_r.name("identifier");
        identifier_r
          %= (lexeme[char_("a-zA-Z") 
                     >> *char_("a-zA-Z0-9_.")]);

        args_r.name("function argument expressions");
        args_r 
          %= lit('(') 
          >> (expression_r % ',')
          > lit(')');

        dims_r.name("array dimensions");
        dims_r 
          %= lit('[') 
          > (expression_r 
             [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
             % ',')
          > lit(']')
          ;
 
        variable_r.name("variable expression");
        variable_r
          %= identifier_r;

      }

      variable_map& var_map_;
      std::stringstream& error_msgs_;


      boost::spirit::qi::rule<Iterator, std::vector<expression>(), 
               whitespace_grammar<Iterator> > 
      args_r;

      boost::spirit::qi::rule<Iterator, std::vector<expression>(), 
               whitespace_grammar<Iterator> > 
      dims_r;

      boost::spirit::qi::rule<Iterator, double_literal(),
                              whitespace_grammar<Iterator> > 
      double_literal_r;

      boost::spirit::qi::rule<Iterator, expression(), 
                              whitespace_grammar<Iterator> > 
      expression_r;

      boost::spirit::qi::rule<Iterator, 
                              boost::spirit::qi::locals<bool>, 
               expression(), whitespace_grammar<Iterator> > 
      factor_r;

      boost::spirit::qi::rule<Iterator, fun(), whitespace_grammar<Iterator> > 
      fun_r;

      boost::spirit::qi::rule<Iterator, std::string(), 
                              whitespace_grammar<Iterator> > 
      identifier_r;

      boost::spirit::qi::rule<Iterator, expression(), 
                              whitespace_grammar<Iterator> > 
      indexed_factor_r;

      // two of these because of type-coercion from index_op to expression
      boost::spirit::qi::rule<Iterator, index_op(), 
                              whitespace_grammar<Iterator> > 
      indexed_factor_2_r; 

      boost::spirit::qi::rule<Iterator, int_literal(), 
                              whitespace_grammar<Iterator> > 
      int_literal_r;

      boost::spirit::qi::rule<Iterator, expression(), 
                              whitespace_grammar<Iterator> > 
      negated_factor_r;

      boost::spirit::qi::rule<Iterator, expression(), 
                              whitespace_grammar<Iterator> > 
      term_r;

      boost::spirit::qi::rule<Iterator, variable(), 
                              whitespace_grammar<Iterator> > 
      variable_r;

    };

    template <typename Iterator>
    struct var_decl_grammar 
      : boost::spirit::qi::grammar<Iterator,
                                   boost::spirit::qi::locals<bool>,
                                   var_decl(bool,var_origin),
                                   whitespace_grammar<Iterator> > {


 
      var_decl_grammar(variable_map& var_map,
                       std::stringstream& error_msgs)
        : var_decl_grammar::base_type(var_decl_r),
          var_map_(var_map),
          error_msgs_(error_msgs),
          expression_g(var_map,error_msgs) {

        using boost::spirit::qi::_1;
        using boost::spirit::qi::char_;
        using boost::spirit::qi::eps;
        using boost::spirit::qi::lexeme;
        using boost::spirit::qi::lit;
        using boost::spirit::qi::_pass;
        using boost::spirit::qi::_val;
        using boost::spirit::qi::labels::_a;
        using boost::spirit::qi::labels::_r1;
        using boost::spirit::qi::labels::_r2;

        // _a = error state local, _r1 constraints allowed inherited
        var_decl_r.name("variable declaration");
        var_decl_r 
          %= (int_decl_r             
              [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | double_decl_r        
                [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | vector_decl_r        
                [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | row_vector_decl_r    
                [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | matrix_decl_r        
                [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | simplex_decl_r       
                [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | pos_ordered_decl_r   
                [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | corr_matrix_decl_r   
                [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              | cov_matrix_decl_r    
                [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2)]
              )
          > eps
            [_pass 
             = validate_decl_constraints_f(_r1,_a,_val,
                                           boost::phoenix::ref(error_msgs_))]
          ;

        int_decl_r.name("integer declaration");
        int_decl_r 
          %= lit("int")
          > -range_brackets_int_r
          > identifier_r 
          > opt_dims_r
          > lit(';');

        double_decl_r.name("double declaration");
        double_decl_r 
          %= lit("double")
          > -range_brackets_double_r
          > identifier_r
          > opt_dims_r
          > lit(';');

        vector_decl_r.name("vector declaration");
        vector_decl_r 
          %= lit("vector")
          > lit('(')
          > expression_g
            [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > lit(')')
          > identifier_r 
          > opt_dims_r
          > lit(';');

        row_vector_decl_r.name("row vector declaration");
        row_vector_decl_r 
          %= lit("row_vector")
          > lit('(')
          > expression_g
            [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > lit(')')
          > identifier_r 
          > opt_dims_r
          > lit(';');

        matrix_decl_r.name("matrix declaration");
        matrix_decl_r 
          %= lit("matrix")
          > lit('(')
          > expression_g
            [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > lit(',')
          > expression_g
            [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > lit(')')
          > identifier_r 
          > opt_dims_r
          > lit(';');

        simplex_decl_r.name("simplex declaration");
        simplex_decl_r 
          %= lit("simplex")
          > lit('(')
          > expression_g
            [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > lit(')')
          > identifier_r 
          > opt_dims_r
          > lit(';'); 

        pos_ordered_decl_r.name("positive ordered declaration");
        pos_ordered_decl_r 
          %= lit("pos_ordered")
          > lit('(')
          > expression_g
            [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > lit(')')
          > identifier_r 
          > opt_dims_r
          > lit(';');

        corr_matrix_decl_r.name("correlation matrix declaration");
        corr_matrix_decl_r 
          %= lit("corr_matrix")
          > lit('(')
          > expression_g
            [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > lit(')')
          > identifier_r 
          > opt_dims_r
          > lit(';');

        cov_matrix_decl_r.name("covariance matrix declaration");
        cov_matrix_decl_r 
          %= lit("cov_matrix")
          > lit('(')
          > expression_g
            [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          > lit(')')
          > identifier_r 
          > opt_dims_r
          > lit(';');

        opt_dims_r.name("array dimensions (optional)");
        opt_dims_r 
          %=  - dims_r;

        dims_r.name("array dimensions");
        dims_r 
          %= lit('[') 
          > (expression_g
             [_pass = validate_int_data_expr_f(_1,
                                               boost::phoenix::ref(var_map_),
                                               boost::phoenix::ref(error_msgs_))]
             % ',')
          > lit(']')
          ;

        range_brackets_int_r.name("range expression pair, brackets");
        range_brackets_int_r 
          %= lit('(') 
          > -(expression_g
             [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))])
          > lit(',')
          > -(expression_g
             [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))])
          > lit(')');

        range_brackets_double_r.name("range expression pair, brackets");
        range_brackets_double_r 
          %= lit('(') 
          > -(expression_g
          [_pass = validate_double_expr_f(_1,boost::phoenix::ref(error_msgs_))])
          > lit(',')
          > -(expression_g
              [_pass 
                = validate_double_expr_f(_1,boost::phoenix::ref(error_msgs_))])
          > lit(')');

        identifier_r.name("identifier");
        identifier_r
          %= (lexeme[char_("a-zA-Z") 
                        >> *char_("a-zA-Z0-9_.")]);

        range_r.name("range expression pair, colon");
        range_r 
          %= expression_g
             [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          >> lit(':') 
          >> expression_g
             [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))];

      }
      
      // global info for parses
      variable_map& var_map_;
      std::stringstream& error_msgs_;

      // grammars
      expression_grammar<Iterator> expression_g;      

      // rules

      boost::spirit::qi::rule<Iterator, corr_matrix_var_decl(), 
                              whitespace_grammar<Iterator> >
      corr_matrix_decl_r;

      boost::spirit::qi::rule<Iterator, cov_matrix_var_decl(), 
                              whitespace_grammar<Iterator> > 
      cov_matrix_decl_r;

      boost::spirit::qi::rule<Iterator, std::vector<expression>(), 
               whitespace_grammar<Iterator> > 
      dims_r;

      boost::spirit::qi::rule<Iterator, double_var_decl(), 
                              whitespace_grammar<Iterator> > 
      double_decl_r;

      boost::spirit::qi::rule<Iterator, std::string(), 
                              whitespace_grammar<Iterator> > 
      identifier_r;

      boost::spirit::qi::rule<Iterator, int_var_decl(), 
                              whitespace_grammar<Iterator> > 
      int_decl_r;

      boost::spirit::qi::rule<Iterator, matrix_var_decl(), 
                              whitespace_grammar<Iterator> > 
      matrix_decl_r;

      boost::spirit::qi::rule<Iterator, std::vector<expression>(),
               whitespace_grammar<Iterator> > 
      opt_dims_r;

      boost::spirit::qi::rule<Iterator, pos_ordered_var_decl(), 
                              whitespace_grammar<Iterator> > 
      pos_ordered_decl_r;

      boost::spirit::qi::rule<Iterator, range(),
                              whitespace_grammar<Iterator> > 
      range_brackets_double_r;

      boost::spirit::qi::rule<Iterator, range(), 
                              whitespace_grammar<Iterator> > 
      range_brackets_int_r;

      boost::spirit::qi::rule<Iterator, range(), 
                              whitespace_grammar<Iterator> > 
      range_r;

      boost::spirit::qi::rule<Iterator, row_vector_var_decl(), 
                              whitespace_grammar<Iterator> > 
      row_vector_decl_r;

      boost::spirit::qi::rule<Iterator, simplex_var_decl(), 
                              whitespace_grammar<Iterator> > 
      simplex_decl_r;

      boost::spirit::qi::rule<Iterator, vector_var_decl(),
                              whitespace_grammar<Iterator> > 
      vector_decl_r;

      boost::spirit::qi::rule<Iterator, 
                              boost::spirit::qi::locals<bool>, 
                              var_decl(bool,var_origin), 
               whitespace_grammar<Iterator> > 
      var_decl_r;

    };

    template <typename Iterator>
    struct statement_grammar 
      : boost::spirit::qi::grammar<Iterator,
                                   statement(bool,var_origin),
                                   whitespace_grammar<Iterator> > {

  
      
      statement_grammar(variable_map& var_map,
                        std::stringstream& error_msgs)
        : statement_grammar::base_type(statement_r),
          var_map_(var_map),
          error_msgs_(error_msgs),
          expression_g(var_map,error_msgs),
          var_decl_g(var_map,error_msgs) {

        using boost::spirit::qi::_1;
        using boost::spirit::qi::char_;
        using boost::spirit::qi::eps;
        using boost::spirit::qi::lexeme;
        using boost::spirit::qi::lit;
        using boost::spirit::qi::_pass;
        using boost::spirit::qi::_val;

        using boost::spirit::qi::labels::_a;
        using boost::spirit::qi::labels::_r1;
        using boost::spirit::qi::labels::_r2;

        // _r1 true if sample_r allowed (inherited)
        // _r2 source of variables allowed for assignments
        // set to true if sample_r are allowed
        statement_r.name("statement");
        statement_r
          %= statement_seq_r(_r1,_r2)
          | for_statement_r(_r1,_r2)
          | assignment_r 
            [_pass 
             = validate_assignment_f(_1,_r2,boost::phoenix::ref(var_map_),
                                     boost::phoenix::ref(error_msgs_))]
          | sample_r(_r1) [_pass = validate_sample_f(_1)]
          | no_op_statement_r
          ;

        // _r1, _r2 same as statement_r
        statement_seq_r.name("sequence of statements");
        statement_seq_r
          %= lit('{')
          > local_var_decls_r[_a = _1]
          > *statement_r(_r1,_r2)
          > lit('}')
          > eps[unscope_locals_f(_a,boost::phoenix::ref(var_map_))]
          ;

        local_var_decls_r
          %= *var_decl_g(false,local_origin); // - constants


        // _r1, _r2 same as statement_r
        for_statement_r.name("for statement");
        for_statement_r
          %= lit("for")
          > lit('(')
          > identifier_r [_pass 
                          = add_loop_identifier_f(_1,_a,
                                         boost::phoenix::ref(var_map_),
                                         boost::phoenix::ref(error_msgs_))]
          > lit("in")
          > range_r
          > lit(')')
          > statement_r(_r1,_r2)
          > eps 
            [remove_loop_identifier_f(_a,boost::phoenix::ref(var_map_))];
          ;

        identifier_r.name("identifier");
        identifier_r
          %= (lexeme[char_("a-zA-Z") 
                     >> *char_("a-zA-Z0-9_.")]);

        range_r.name("range expression pair, colon");
        range_r 
          %= expression_g
             [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
          >> lit(':') 
          >> expression_g
             [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))];

        assignment_r.name("variable assignment by expression");
        assignment_r
          %= var_lhs_r
          >> lit("<-")
          > expression_g
          > lit(';') 
          ;

        var_lhs_r.name("variable and array dimensions");
        var_lhs_r 
          %= identifier_r 
          >> opt_dims_r;

        opt_dims_r.name("array dimensions (optional)");
        opt_dims_r 
          %=  - dims_r;

        dims_r.name("array dimensions");
        dims_r 
          %= lit('[') 
          > (expression_g
             [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
             % ',')
          > lit(']')
          ;

        // inherited  _r1 = true if samples allowed as statements
        sample_r.name("distribution of expression");
        sample_r 
          %= expression_g
          >> lit('~')
          > eps
           [_pass 
            = validate_allow_sample_f(_r1,boost::phoenix::ref(error_msgs_))] 
          > distribution_r
          > -truncation_range_r
          > lit(';');

        distribution_r.name("distribution and parameters");
        distribution_r
          %= identifier_r
          >> lit('(')
          >> -(expression_g % ',')
          > lit(')');

        truncation_range_r.name("range pair");
        truncation_range_r
          %= lit('T')
          > lit('(') 
          > -expression_g
          > lit(',')
          > -expression_g
          > lit(')');

        no_op_statement_r.name("no op statement");
        no_op_statement_r 
          %= lit(';') [_val = no_op_statement()];  // ok to re-use instance

      }



      // global info for parses
      variable_map& var_map_;
      std::stringstream& error_msgs_;
      
      // grammars
      expression_grammar<Iterator> expression_g;  
      var_decl_grammar<Iterator> var_decl_g;

      // rules
      boost::spirit::qi::rule<Iterator, assignment(), 
                              whitespace_grammar<Iterator> > 
      assignment_r;

      boost::spirit::qi::rule<Iterator, std::vector<expression>(), 
                              whitespace_grammar<Iterator> > 
      dims_r;

      boost::spirit::qi::rule<Iterator, distribution(),
                              whitespace_grammar<Iterator> >
      distribution_r;

      boost::spirit::qi::rule<Iterator, 
                              boost::spirit::qi::locals<std::string>, 
                              for_statement(bool,var_origin), 
                              whitespace_grammar<Iterator> > 
      for_statement_r;

      boost::spirit::qi::rule<Iterator, std::string(), 
                              whitespace_grammar<Iterator> > 
      identifier_r;

      boost::spirit::qi::rule<Iterator, std::vector<var_decl>(), 
                              whitespace_grammar<Iterator> >
      local_var_decls_r;

      boost::spirit::qi::rule<Iterator, no_op_statement(), 
                              whitespace_grammar<Iterator> > 
      no_op_statement_r;

      boost::spirit::qi::rule<Iterator, std::vector<expression>(),
                              whitespace_grammar<Iterator> > 
      opt_dims_r;

      boost::spirit::qi::rule<Iterator, range(), 
                              whitespace_grammar<Iterator> > 
      range_r;

      boost::spirit::qi::rule<Iterator, sample(bool),
                              whitespace_grammar<Iterator> > 
      sample_r;

      boost::spirit::qi::rule<Iterator, 
                              statement(bool,var_origin), 
                              whitespace_grammar<Iterator> > 
      statement_r;

      boost::spirit::qi::rule<Iterator, 
                              boost::spirit::qi::locals<std::vector<var_decl> >,
                              statements(bool,var_origin), 
                              whitespace_grammar<Iterator> >
      statement_seq_r;

      boost::spirit::qi::rule<Iterator, range(), 
                              whitespace_grammar<Iterator> > 
      truncation_range_r;

      boost::spirit::qi::rule<Iterator, variable_dims(),
                              whitespace_grammar<Iterator> > 
      var_lhs_r;

    };
                               
    template <typename Iterator>
    struct program_grammar 
      : boost::spirit::qi::grammar<Iterator, 
                                   program(), 
                                   whitespace_grammar<Iterator> > {

      program_grammar() 
        : program_grammar::base_type(program_r),
          var_map_(),
          error_msgs_(),
          expression_g(var_map_,error_msgs_),
          var_decl_g(var_map_,error_msgs_),
          statement_g(var_map_,error_msgs_) {

        using boost::spirit::qi::eps;
        using boost::spirit::qi::lit;

        program_r.name("program");
        program_r 
          %= -data_var_decls_r
          > -derived_data_var_decls_r
          > -param_var_decls_r
          // scope lp__ to "transformed params" and "model" only
          > eps[add_lp_var_f(boost::phoenix::ref(var_map_))]
          > -derived_var_decls_r
          > model_r
          > eps[remove_lp_var_f(boost::phoenix::ref(var_map_))]
          > -generated_var_decls_r
          ;

        model_r.name("model declaration");
        model_r 
          %= lit("model")
          > statement_g(true,local_origin)  // assign only to locals
          ;

        data_var_decls_r.name("data variable declarations");
        data_var_decls_r
          %= lit("data")
          > lit('{')
          > *var_decl_g(true,data_origin) // +constraints
          > lit('}');

        derived_data_var_decls_r.name("transformed data block");
        derived_data_var_decls_r
          %= lit("transformed")
          >> lit("data")
          > lit('{')
          > *var_decl_g(true,transformed_data_origin)  // -constraints
          > *statement_g(false,transformed_data_origin) // -sampling
          > lit('}');

        param_var_decls_r.name("parameter variable declarations");
        param_var_decls_r
          %= lit("parameters")
          > lit('{')
          > *var_decl_g(true,parameter_origin) // +constraints
          > lit('}');

        derived_var_decls_r.name("derived variable declarations");
        derived_var_decls_r
          %= lit("transformed")
          >> lit("parameters")
          > lit('{')
          > *var_decl_g(true,transformed_parameter_origin) // -constraints
          > *statement_g(false,transformed_parameter_origin) // -sampling
          > lit('}');

        generated_var_decls_r.name("generated variable declarations");
        generated_var_decls_r
          %= lit("generated")
          > lit("quantities")
          > lit('{')
          > *var_decl_g(true,derived_origin) // -constraints
          > *statement_g(false,derived_origin) // -sampling
          > lit('}');

        using boost::spirit::qi::on_error;
        using boost::spirit::qi::rethrow;
        on_error<rethrow>(program_r,
                          (std::ostream&)error_msgs_
                          << std::endl
                          << boost::phoenix::val("Parser expecting: ")
                          << boost::spirit::qi::labels::_4); 
      }


      // global info for parses
      variable_map var_map_;
      std::stringstream error_msgs_;

      // grammars
      expression_grammar<Iterator> expression_g;
      var_decl_grammar<Iterator> var_decl_g;
      statement_grammar<Iterator> statement_g;

      // rules

      boost::spirit::qi::rule<Iterator, std::vector<var_decl>(), 
               whitespace_grammar<Iterator> >       
      data_var_decls_r;

      boost::spirit::qi::rule<Iterator, std::pair<std::vector<var_decl>,
                                   std::vector<statement> >(), 
               whitespace_grammar<Iterator> > 
      derived_data_var_decls_r;

      boost::spirit::qi::rule<Iterator, std::pair<std::vector<var_decl>,
                                   std::vector<statement> >(), 
               whitespace_grammar<Iterator> > 
      derived_var_decls_r;

      boost::spirit::qi::rule<Iterator, std::pair<std::vector<var_decl>,
                                   std::vector<statement> >(), 
               whitespace_grammar<Iterator> > 
      generated_var_decls_r;

      boost::spirit::qi::rule<Iterator, statement(), 
                              whitespace_grammar<Iterator> > 
      model_r;

      boost::spirit::qi::rule<Iterator, std::vector<var_decl>(), 
               whitespace_grammar<Iterator> >
      param_var_decls_r;

      boost::spirit::qi::rule<Iterator, program(),
                              whitespace_grammar<Iterator> >
      program_r;
    
    };

    // Cut and paste source for iterator & reporting pattern:
    // http://boost-spirit.com/home/articles/qi-example
    //                 /tracking-the-input-position-while-parsing/
    // http://boost-spirit.com/dl_more/parsing_tracking_position
    //                 /stream_iterator_errorposition_parsing.cpp
    bool parse(std::istream& input, 
               const std::string& filename, 
               program& result) {

      using boost::spirit::classic::position_iterator2;
      using boost::spirit::multi_pass;
      using boost::spirit::make_default_multi_pass;
      using std::istreambuf_iterator;

      using boost::spirit::qi::expectation_failure;
      using boost::spirit::classic::file_position_base;
      using boost::spirit::qi::phrase_parse;


      // iterate over stream input
      typedef istreambuf_iterator<char> base_iterator_type;
      typedef multi_pass<base_iterator_type>  forward_iterator_type;
      typedef position_iterator2<forward_iterator_type> pos_iterator_type;

      base_iterator_type in_begin(input);
      
      forward_iterator_type fwd_begin = make_default_multi_pass(in_begin);
      forward_iterator_type fwd_end;
      
      pos_iterator_type position_begin(fwd_begin, fwd_end, filename);
      pos_iterator_type position_end;
      
      program_grammar<pos_iterator_type> prog_grammar;
      whitespace_grammar<pos_iterator_type> whitesp_grammar;
      
      bool parse_succeeded = false;
      try {
        parse_succeeded = phrase_parse(position_begin, 
                                       position_end,
                                       prog_grammar,
                                       whitesp_grammar,
                                       result);
      } catch (const expectation_failure<pos_iterator_type>& e) {
        const file_position_base<std::string>& pos = e.first.get_position();
        std::stringstream msg;
        msg << "LOCATION:  file=" << pos.file
            << "; line=" << pos.line 
            << ", colum=" << pos.column 
            << std::endl;
        msg << std::endl << e.first.get_currentline() 
            << std::endl;
        for (int i = 2; i < pos.column; ++i)
          msg << ' ';
        msg << " ^-- here" 
            << std::endl << std::endl;

        msg << "DIAGNOSTIC(S) FROM PARSER:" << std::endl;
        msg << prog_grammar.error_msgs_.str() << std::endl << std::endl;
        throw std::invalid_argument(msg.str());

      } catch (const std::runtime_error& e) {
        std::stringstream msg;
        msg << "LOCATION: unknown" << std::endl;

        msg << "DIAGNOSTICS FROM PARSER:" << std::endl;
        msg << prog_grammar.error_msgs_.str() << std::endl << std::endl;
        throw std::invalid_argument(msg.str());
      }
      
      bool consumed_all_input = (position_begin == position_end); 
      bool success = parse_succeeded && consumed_all_input;

      if (!success) {      
        std::stringstream msg;
        if (!parse_succeeded)
          msg << "PARSE DID NOT SUCCEED." << std::endl; 
        if (!consumed_all_input)
          msg << "ERROR: non-whitespace beyond end of program:" << std::endl;
        
        const file_position_base<std::string>& pos 
          = position_begin.get_position();
        msg << "LOCATION: file=" << pos.file
            << "; line=" << pos.line
            << ", column=" << pos.column
            << std::endl;
        msg << position_begin.get_currentline() 
            << std::endl;
        for (int i = 2; i < pos.column; ++i)
          msg << ' ';
        msg << " ^-- starting here" 
            << std::endl << std::endl;

        msg << "DIAGNOSTICS FROM PARSER:" << std::endl;
        msg << prog_grammar.error_msgs_.str() << std::endl << std::endl;

        throw std::invalid_argument(msg.str());
      }
      return true;
    }

  }

}

#endif
