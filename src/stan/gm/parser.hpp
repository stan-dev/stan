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
#include <sstream>
#include <string>
#include <utility>
#include <vector>

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

BOOST_FUSION_ADAPT_STRUCT(stan::gm::var,
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
			  (stan::gm::statement, statement_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::distribution,
			  (std::string, family_)
			  (std::vector<stan::gm::expression>, args_) )


BOOST_FUSION_ADAPT_STRUCT(stan::gm::statements,
			  (std::vector<stan::gm::statement>, statements_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::sample,
			  (stan::gm::expression, expr_)
			  (stan::gm::distribution, dist_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::assignment,
			  (stan::gm::var, var_)
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

    struct validate_expr_type {
      template <typename T>
      struct result { typedef bool type; };

      bool operator()(const expression& expr) const {
	// std::cout << "validating expr type=" << expr.expression_type() << std::endl;
	return !expr.expression_type().is_ill_formed();
      }
    };
    boost::phoenix::function<validate_expr_type> validate_expr_type_f;

    struct validate_primitive_int_type {
      template <typename T>
      struct result { typedef bool type; };

      bool operator()(const expression& expr) const {
	return expr.expression_type().is_primitive_int();
      }
    };
    boost::phoenix::function<validate_primitive_int_type> validate_primitive_int_type_f;

    // the following is for debugging:
    // void generate_expression(const expression& e, std::ostream& o);
    // struct print_expr {
    //   template <typename T1, typename T2>
    //   struct result { typedef void type; };

    //   void operator()(const expression& e, const std::string& s) const {
    // 	std::cout << "***print_expression " << s << std::endl;
    // 	std::cout << "    expr=";
    // 	generate_expression(e,std::cout);
    // 	std::cout << std::endl;
    // 	std::cout << "    type=" << e.expression_type() << std::endl;
    //   }
    // };
    // boost::phoenix::function<print_expr> print;

    struct set_variable_type {
      template <typename T1, typename T2>
      struct result { typedef T1 type; };
      template <typename T>
      T operator()(T& var_expr, std::map<std::string,base_var_decl>& name_to_type) const {
	if (name_to_type.find(var_expr.name_) == name_to_type.end()) {
	  // FIXME: fail
	}
	var_expr.set_type(name_to_type[var_expr.name_].base_type_, 
			  name_to_type[var_expr.name_].dims_.size());
	return var_expr;
      }
    };
    boost::phoenix::function<set_variable_type> set_var_type;

    struct add_var_decl {
      template <typename T1, typename T2, typename T3>
      struct result { typedef T1 type; };
      template <typename T>
      T operator()(const T& var_decl, std::map<std::string,base_var_decl>& name_to_type, bool& pass) const {
	if (name_to_type.find(var_decl.name_) != name_to_type.end()) {
	  // variable already exists
	  pass = false;
	  return var_decl;
	}
	pass = true;
	name_to_type[var_decl.name_] = var_decl;
	return var_decl;
      }
    };
    boost::phoenix::function<add_var_decl> add_var;

    struct add_loop_identifier {
      template <typename T1, typename T2, typename T3>
      struct result { typedef bool type; };
      bool operator()(const std::string& name, 
		      std::string& name_local,
		      std::map<std::string,base_var_decl>& name_to_type) const {
	name_local = name;
	if (name_to_type.find(name) != name_to_type.end())
	  return false; // variable exists
	name_to_type[name] = base_var_decl(name,std::vector<expression>(),INT_T);
	return true;
      }
    };
    boost::phoenix::function<add_loop_identifier> add_loop_identifier_f;

    struct remove_loop_identifier {
      template <typename T1, typename T2>
      struct result { typedef void type; };
      void operator()(const std::string& name, 
		      std::map<std::string,base_var_decl>& name_to_type) const {
	name_to_type.erase(name);
      }
    };
    boost::phoenix::function<remove_loop_identifier> remove_loop_identifier_f;


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
      std::map<std::string,base_var_decl> var_name_to_decl_;
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
	  = -data_var_decls_r
	  > -derived_data_var_decls_r
	  > -param_var_decls_r
	  > -derived_var_decls_r
	  > model_r;
	
	model_r.name("model declaration");
	model_r 
	  = qi::lit("model")
	  > statement_r;

	data_var_decls_r.name("data variable declarations");
	data_var_decls_r
	  = qi::lit("data")
	  > qi::lit('{')
	  > *var_decl_r
	  > qi::lit('}');

	derived_data_var_decls_r.name("derived data variable declaration and statement");
	derived_data_var_decls_r
	  = qi::lit("derived")
	  >> qi::lit("data")
	  > qi::lit('{')
	  > *var_decl_r
	  > *statement_r
	  > qi::lit('}');

	param_var_decls_r.name("parameter variable declarations");
	param_var_decls_r
	  = qi::lit("parameters")
	  > qi::lit('{')
	  > *var_decl_r
	  > qi::lit('}');

	derived_var_decls_r.name("derived variable declarations");
	derived_var_decls_r
	  = qi::lit("derived")
	  >> qi::lit("parameters")
	  > qi::lit('{')
	  > *var_decl_r
	  > *statement_r
	  > qi::lit('}');

	// duplication because top-level is variant, not specific
	var_decl_r.name("variable declaration");
	var_decl_r 
	  %= (int_decl_r             [_val = add_var(_1,boost::phoenix::ref(var_name_to_decl_),_a)]
	      | double_decl_r        [_val = add_var(_1,boost::phoenix::ref(var_name_to_decl_),_a)]
	      | vector_decl_r        [_val = add_var(_1,boost::phoenix::ref(var_name_to_decl_),_a)]
	      | row_vector_decl_r    [_val = add_var(_1,boost::phoenix::ref(var_name_to_decl_),_a)]
	      | matrix_decl_r        [_val = add_var(_1,boost::phoenix::ref(var_name_to_decl_),_a)]
	      | simplex_decl_r       [_val = add_var(_1,boost::phoenix::ref(var_name_to_decl_),_a)]
	      | pos_ordered_decl_r   [_val = add_var(_1,boost::phoenix::ref(var_name_to_decl_),_a)]
	      | corr_matrix_decl_r   [_val = add_var(_1,boost::phoenix::ref(var_name_to_decl_),_a)]
	      | cov_matrix_decl_r    [_val = add_var(_1,boost::phoenix::ref(var_name_to_decl_),_a)]
	      )
	  > qi::eps[_pass = _a] // hack to get error message out
	  ;
	
	int_decl_r.name("integer declaration");
	int_decl_r 
	  = qi::lit("int")
	  > -range_brackets_r
	  > identifier_r 
	  > opt_dims_r
	  > qi::lit(';');

	double_decl_r.name("double declaration");
	double_decl_r 
	  = qi::lit("double")
	  > -range_brackets_r
	  > identifier_r
	  > opt_dims_r
	  > qi::lit(';');

	vector_decl_r.name("vector declaration");
	vector_decl_r 
	  = qi::lit("vector")
	  > qi::lit('(')
	  > expression_r
	  > qi::lit(')')
	  > identifier_r 
	  > opt_dims_r
	  > qi::lit(';');

	row_vector_decl_r.name("row vector declaration");
	row_vector_decl_r 
	  = qi::lit("row_vector")
	  > qi::lit('(')
	  > expression_r
	  > qi::lit(')')
	  > identifier_r 
	  > opt_dims_r
	  > qi::lit(';');

	matrix_decl_r.name("matrix declaration");
	matrix_decl_r 
	  = qi::lit("matrix")
	  > qi::lit('(')
	  > expression_r
	  > qi::lit(',')
	  > expression_r
	  > qi::lit(')')
	  > identifier_r 
	  > opt_dims_r
	  > qi::lit(';');

	simplex_decl_r.name("simplex declaration");
	simplex_decl_r 
	  = qi::lit("simplex")
	  > qi::lit('(')
	  > expression_r
	  > qi::lit(')')
	  > identifier_r 
	  > opt_dims_r
	  > qi::lit(';');

	pos_ordered_decl_r.name("positive ordered declaration");
	pos_ordered_decl_r 
	  = qi::lit("pos_ordered")
	  > qi::lit('(')
	  > expression_r
	  > qi::lit(')')
	  > identifier_r 
	  > opt_dims_r
	  > qi::lit(';');

	cov_matrix_decl_r.name("positive definite symmetric (covariance) matrix declaration");
	cov_matrix_decl_r 
	  = qi::lit("cov_matrix")
	  > qi::lit('(')
	  > expression_r
	  > qi::lit(')')
	  > identifier_r 
	  > opt_dims_r
	  > qi::lit(';');

	corr_matrix_decl_r.name("correlation matrix declaration");
	corr_matrix_decl_r 
	  = qi::lit("corr_matrix")
	  > qi::lit('(')
	  > expression_r
	  > qi::lit(')')
	  > identifier_r 
	  > opt_dims_r
	  > qi::lit(';');

	expression_r.name("expression");
	expression_r 
	  %=  term_r                                 [_val = _1]
	  >> *( (qi::lit('+') > term_r         [_val += _1])
		|   (qi::lit('-') > term_r    [_val -= _1])
		)
	  // > qi::eps[_pass = validate_expr_type_f(_val)];
	  ;

	term_r.name("term");
	term_r 
	  %= ( indexed_factor_r                          [_val = _1]
	      >> *( (qi::lit('*') > indexed_factor_r     [_val *= _1])
		    | (qi::lit('/') > indexed_factor_r   [_val /= _1])
		    )
	      )
	  ;

	indexed_factor_r.name("(optionally) indexed factor [sub]");
	indexed_factor_r
	  %= (factor_r >> (*dims_r));
	
	factor_r.name("factor");
	factor_r
	  %= int_literal_r               [_val = _1]
	  | double_literal_r             [_val = _1]
	  | fun_r                        [_val = _1] 
	  | variable_r                   [_val = set_var_type(_1,boost::phoenix::ref(var_name_to_decl_))]
	  | ( qi::lit('(') 
	      > expression_r             [_val = _1] 
	      > qi::lit(')') )
	  | ( qi::lit('-') 
	      > factor_r                 [_val = neg(_1)] 
	      )
	  | ( qi::lit('+') 
	      > factor_r                 [_val = _1]
	      )
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
	  = identifier_r;

	fun_r.name("function and argument expressions");
	fun_r 
	  = identifier_r 
	  >> args_r; 
	    
	var_r.name("variable and array dimensions");
	var_r 
	  = identifier_r 
	  >> opt_dims_r;

	opt_dims_r.name("array dimensions (optional)");
	opt_dims_r 
	  =  - dims_r;

	dims_r.name("array dimensions");
	dims_r 
	  =  ( qi::lit('[') 
	       >> (expression_r % ',')
	       > qi::lit(']') );
	
	range_r.name("range expression pair, colon");
	range_r 
	  = expression_r 
	  >> qi::lit(':') 
	  >> expression_r;

	range_brackets_r.name("range expression pair, brackets");
	range_brackets_r 
	  = qi::lit('(') 
	  >> -expression_r
	  > qi::lit(',')
	  >> -expression_r
	  > qi::lit(')');

	args_r.name("function argument expressions");
	args_r 
	  = qi::lit('(') 
	  >> (expression_r % ',')
	  > qi::lit(')');

	identifier_r.name("identifier");
	identifier_r
	  = (qi::lexeme[qi::char_("a-zA-Z") 
			>> *qi::char_("a-zA-Z0-9_.")]);

	distribution_r.name("distribution and parameters");
	distribution_r
	  = identifier_r
	  >> qi::lit('(')
	  >> -(expression_r % ',')
	  > qi::lit(')');

	sample_r.name("distribution of expression");
	sample_r 
	  = expression_r
	  >> qi::lit('~')
	  > distribution_r
	  > qi::lit(';');

	assignment_r.name("variable assignment by expression");
	assignment_r
	  = var_r
	  >> qi::lit("<-")
	  > expression_r
	  > qi::lit(';');

	statement_r.name("statement");
	statement_r
	  = statement_seq_r
	  | for_statement_r
	  | assignment_r
	  | sample_r;

	for_statement_r.name("for statement");
	for_statement_r
	  %= qi::lit("for")
	  > qi::lit('(')
	  > identifier_r [_pass = add_loop_identifier_f(_1,_a,boost::phoenix::ref(var_name_to_decl_))]
	  > qi::lit("in")
	  > range_r
	  > qi::lit(')')
	  > statement_r
	  > qi::eps [remove_loop_identifier_f(_a,boost::phoenix::ref(var_name_to_decl_))];
	  ;

	statement_seq_r.name("sequence of statements");
	statement_seq_r
	  = qi::lit('{')
	  >> *statement_r
	  > qi::lit('}');

	qi::on_error<qi::rethrow>(var_decl_r,
				  std::cerr 
				  << boost::phoenix::val("ERROR: Duplicate variable declaration.")
				  << std::endl);

	qi::on_error<qi::rethrow>(program_r,
				  std::cerr << boost::phoenix::val("ERROR: Expected ")
				  << _4 
				  << std::endl);
      }

      qi::rule<Iterator, expression(), whitespace_grammar<Iterator> > expression_r;
      qi::rule<Iterator, expression(), whitespace_grammar<Iterator> > term_r;
      qi::rule<Iterator, expression(), whitespace_grammar<Iterator> > factor_r;
      qi::rule<Iterator, variable(), whitespace_grammar<Iterator> > variable_r;
      qi::rule<Iterator, int_literal(), whitespace_grammar<Iterator> > int_literal_r;
      qi::rule<Iterator, double_literal(), whitespace_grammar<Iterator> > double_literal_r;
      qi::rule<Iterator, var(), whitespace_grammar<Iterator> > var_r;
      qi::rule<Iterator, fun(), whitespace_grammar<Iterator> > fun_r;
      qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > identifier_r;
      qi::rule<Iterator, std::vector<expression>(), whitespace_grammar<Iterator> > opt_dims_r;
      qi::rule<Iterator, std::vector<expression>(), whitespace_grammar<Iterator> > dims_r;
      qi::rule<Iterator, range(), whitespace_grammar<Iterator> > range_r;
      qi::rule<Iterator, range(), whitespace_grammar<Iterator> > range_brackets_r;
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
      qi::rule<Iterator, qi::locals<bool>, var_decl(), whitespace_grammar<Iterator> > var_decl_r;
      qi::rule<Iterator, std::vector<var_decl>(), whitespace_grammar<Iterator> > data_var_decls_r;
      qi::rule<Iterator, std::pair<std::vector<var_decl>,std::vector<statement> >(), 
	       whitespace_grammar<Iterator> > derived_data_var_decls_r;
      qi::rule<Iterator, std::vector<var_decl>(), whitespace_grammar<Iterator> > param_var_decls_r;
      qi::rule<Iterator, std::pair<std::vector<var_decl>,std::vector<statement> >(), 
	       whitespace_grammar<Iterator> > derived_var_decls_r;
      qi::rule<Iterator, program(), whitespace_grammar<Iterator> > program_r;
      qi::rule<Iterator, distribution(), whitespace_grammar<Iterator> > distribution_r;
      qi::rule<Iterator, sample(), whitespace_grammar<Iterator> > sample_r;
      qi::rule<Iterator, assignment(), whitespace_grammar<Iterator> > assignment_r;
      qi::rule<Iterator, statement(), whitespace_grammar<Iterator> > statement_r;
      qi::rule<Iterator, statements(), whitespace_grammar<Iterator> > statement_seq_r;
      qi::rule<Iterator, qi::locals<std::string>, for_statement(), 
               whitespace_grammar<Iterator> > for_statement_r;
      qi::rule<Iterator, statement(), whitespace_grammar<Iterator> > model_r;
      qi::rule<Iterator, expression(), whitespace_grammar<Iterator> > indexed_factor_r;
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
      
      program_grammar<pos_iterator_type> program_grammar;
      whitespace_grammar<pos_iterator_type> whitespace_grammar;
      
      bool success = 0;
      try {
	success = qi::phrase_parse(position_begin, 
				   position_end,
				   program_grammar,
				   whitespace_grammar,
				   result);
      } catch (const qi::expectation_failure<pos_iterator_type>& e) {
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
	throw std::runtime_error(msg.str());
      }
      return success && (position_begin == position_end); // want to consume ALL input
    }

  }

}

#endif
