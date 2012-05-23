#ifndef __STAN__GM__PARSER__VAR_DECLS_GRAMMAR_DEF__HPP__
#define __STAN__GM__PARSER__VAR_DECLS_GRAMMAR_DEF__HPP__

#include <cstddef>
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

#include <boost/spirit/include/qi.hpp>
// FIXME: get rid of unused include
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>

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

#include <stan/gm/ast.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>
#include <stan/gm/grammars/var_decls_grammar.hpp>
#include <stan/gm/grammars/common_adaptors_def.hpp>

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

BOOST_FUSION_ADAPT_STRUCT(stan::gm::ordered_var_decl,
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

namespace stan {

  namespace gm {

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
      bool operator()(const ordered_var_decl& x) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found ordered." << std::endl;
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
                      << "; declared in block=";
          print_var_origin(error_msgs_,origin);
          error_msgs_ << std::endl;
        }
        return is_data;
      }
      bool operator()(const fun& x) const {
        for (size_t i = 0; i < x.args_.size(); ++i)
          if (!boost::apply_visitor(*this,x.args_[i].expr_))
            return false;
        return true;
      }
      bool operator()(const index_op& x) const {
        if (!boost::apply_visitor(*this,x.expr_.expr_))
          return false;
        for (size_t i = 0; i < x.dimss_.size(); ++i)
          for (size_t j = 0; j < x.dimss_[i].size(); ++j)
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


    struct add_var {
      template <typename T1, typename T2, typename T3, typename T4, typename T5>
      struct result { typedef T1 type; };
      // each type derived from base_var_decl gets own instance
      template <typename T>
      T operator()(const T& var_decl, 
                   variable_map& vm,
                   bool& pass,
                   const var_origin& vo,
                   std::ostream& error_msgs) const {
        if (vm.exists(var_decl.name_)) {
          // variable already exists
          pass = false;
          error_msgs << "variable already declared, name="
                     << var_decl.name_ 
                     << std::endl;
          return var_decl;
        }
        pass = true;  // probably don't need to set true
        vm.add(var_decl.name_,var_decl,vo);
        return var_decl;
      }
    };
    boost::phoenix::function<add_var> add_var_f;

    struct validate_decl_constraints {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef bool type; };

      bool operator()(const bool& allow_constraints,
                      const bool& declaration_ok,
                      const var_decl& var_decl,
                      std::stringstream& error_msgs) const {
        if (!declaration_ok) {
          error_msgs << "Problem with declaration." << std::endl;
          return false; // short-circuits test of constraints
        }
        if (allow_constraints)
          return true;
        validate_no_constraints_vis vis(error_msgs);
        bool constraints_ok = boost::apply_visitor(vis,var_decl.decl_);
        return constraints_ok;
      }
    };
    boost::phoenix::function<validate_decl_constraints> 
    validate_decl_constraints_f;

    struct validate_identifier {
      template <typename T1, typename T2>
      struct result { typedef bool type; };

      bool operator()(const std::string& identifier,
                      std::stringstream& error_msgs) const {
        int len = identifier.size();
        if (len >= 2
            && identifier[len-1] == '_'
            && identifier[len-2] == '_') {
          error_msgs << "identifiers cannot end in double underscore (__)"
                     << "; found identifer=" << identifier;
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_identifier> validate_identifier_f;

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


    template <typename Iterator>
    var_decls_grammar<Iterator>::var_decls_grammar(variable_map& var_map,
                                                   std::stringstream& error_msgs)
      : var_decls_grammar::base_type(var_decls_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        expression_g(var_map,error_msgs)
    {

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

      var_decls_r.name("variable declarations");
      var_decls_r 
        %= *var_decl_r(_r1,_r2);

      // _a = error state local, _r1 constraints allowed inherited
      var_decl_r.name("variable declaration");
      var_decl_r 
        %= (int_decl_r             
            [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs))]
            | double_decl_r        
            [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs_))]
            | vector_decl_r        
            [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs_))]
            | row_vector_decl_r    
            [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs_))]
            | matrix_decl_r        
            [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs_))]
            | simplex_decl_r       
            [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs_))]
            | ordered_decl_r   
            [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs_))]
            | corr_matrix_decl_r   
            [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs_))]
            | cov_matrix_decl_r    
            [_val = add_var_f(_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs_))]
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

      double_decl_r.name("real declaration");
      double_decl_r 
        %= lit("real")
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

      ordered_decl_r.name("positive ordered declaration");
      ordered_decl_r 
        %= lit("ordered")
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
        %= identifier_name_r
          [_pass = validate_identifier_f(_1,boost::phoenix::ref(error_msgs_))]

        ;

      identifier_name_r.name("identifier subrule");
      identifier_name_r
        %= lexeme[char_("a-zA-Z") 
                  >> *char_("a-zA-Z0-9_.")]
        ;
        

      range_r.name("range expression pair, colon");
      range_r 
        %= expression_g
        [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))]
        >> lit(':') 
        >> expression_g
        [_pass = validate_int_expr_f(_1,boost::phoenix::ref(error_msgs_))];

    }


  }
}
#endif
