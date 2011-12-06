#ifndef __STAN__GM__AST_HPP__
#define __STAN__GM__AST_HPP__

#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_fusion.hpp>
#include <boost/spirit/include/phoenix_object.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <boost/variant/recursive_variant.hpp>
#include <boost/variant/static_visitor.hpp>

#include <climits>
#include <iomanip>
#include <iostream>
#include <istream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace stan {

  namespace gm {

    struct nil {
      /* placeholder val for boost::variant default ctors */
    };

    namespace qi = boost::spirit::qi;
    namespace ascii = boost::spirit::ascii;

    // components of abstract syntax tree 
    struct assignment;
    struct binary_op;
    struct distribution;
    struct double_var_decl;
    struct double_literal;
    struct expression;
    struct for_statement;
    struct fun;
    struct identifier;
    struct index_op;
    struct int_literal;
    struct inv_var_decl;
    struct matrix_var_decl;
    struct pos_ordered_var_decl;
    struct program;
    struct range;
    struct row_vector_var_decl;
    struct sample;
    struct simplex_var_decl;
    struct statement;
    struct statements;
    struct unary_op;
    struct var;
    struct variable;
    struct var_decl;
    struct var_type;
    struct vector_var_decl;


    enum base_expr_type {
      INT_T,
      DOUBLE_T,
      VECTOR_T, // includes: SIMPLEX_T, POS_ORDERED_T
      ROW_VECTOR_T,
      MATRIX_T,
      ILL_FORMED_T // includes: CORR_MATRIX_T, COV_MATRIX_T
    };

    class expr_type {
    private: 
      base_expr_type base_type_;
      unsigned int num_dims_;
    public:
      expr_type() 
	: base_type_(ILL_FORMED_T),
	  num_dims_(0) { 
      }
      expr_type(const base_expr_type base_type) 
	: base_type_(base_type),
	  num_dims_(0) {
      }
      expr_type(const base_expr_type base_type,
		unsigned int num_dims) 
	: base_type_(base_type),
	  num_dims_(num_dims) {
      }
      bool operator==(const expr_type& et) const {
	return base_type_ == et.base_type_
	  && num_dims_ == et.num_dims_;
      }
      bool operator!=(const expr_type& et) const {
	return !(*this == et);
      }
      bool is_primitive() const {
	return is_primitive_int() 
	  || is_primitive_double();
      }
      bool is_primitive_int() const {
	return base_type_ == INT_T
	  && num_dims_ == 0U;
      }
      bool is_primitive_double() const {
	return base_type_ == DOUBLE_T
	  && num_dims_ == 0U;
      }
      base_expr_type type() const {
	return base_type_;
      }
      unsigned int num_dims() const {
	return num_dims_;
      }
    };

    std::ostream& operator<<(std::ostream& o, const expr_type& et) {
      switch (et.type()) {
      case INT_T: o << "int"; break;
      case DOUBLE_T: o << "double"; break;
      case VECTOR_T: o << "vector"; break;
      case ROW_VECTOR_T: o << "row_vector"; break;
      case MATRIX_T: o << "matrix"; break;
      case ILL_FORMED_T: o << "ill_formed"; break;
      }
      o << '[' << et.num_dims() << ']';
      return o;
    }

    expr_type promote_primitive(const expr_type& et) {
      if (!et.is_primitive())
	return expr_type();
      return et;
    }

    expr_type promote_primitive(const expr_type& et1,
				const expr_type& et2) {
      if (!et1.is_primitive() || !et2.is_primitive())
	return expr_type();
      return et1.type() == DOUBLE_T ? et1 : et2;
    }

    typedef std::pair<expr_type, std::vector<expr_type> > function_signature_t;

    /**
     * Singleton for all of the function signature specifications.
     * This includes constants which are treated as nullary functions.
     */
    class function_signatures {
    public:

      /**
       * Return the single function signatures object.  
       *
       * @return The single instance.
       */
      static function_signatures& instance() {
	// FIXME:  for threaded models, requires double-check lock
	if (!sigs_)
	  sigs_ = new function_signatures;
	return *sigs_;
      }

      /**
       * Add a signature for the function with specified name,
       * result type, and argument types.
       *
       * @param name Name of function.
       * @param result_type Return type for function.
       * @param arg_types Sequence of argument types for function.
       */
      void add(const std::string& name,
	       const expr_type& result_type,
	       const std::vector<expr_type>& arg_types) {
	sigs_map_[name].push_back(function_signature_t(result_type,arg_types));

      }

      /**
       * Add a signature for the nullary function with specified name
       * and result type.
       *
       * @param name Name of function.
       * @param result_type Return type for function.
       */
      void add(const std::string& name,
	       const expr_type& result_type) {
	std::vector<expr_type> arg_types;
	add(name,result_type,arg_types);
      }

      /**
       * Add a signature for the unary function with specified name,
       * result type and argument type.
       *
       * @param name Name of function.
       * @param result_type Return type for function.
       * @param arg_type Type of argument.
       */
      void add(const std::string& name,
	       const expr_type& result_type,
	       const expr_type& arg_type) {
	std::vector<expr_type> arg_types;
	arg_types.push_back(arg_type);
	add(name,result_type,arg_types);
      }

      /**
       * Add a signature for the binary function with specified name,
       * result type and argument types.
       *
       * @param name Name of function.
       * @param result_type Return type for function.
       * @param arg_type1 Type of first argument.
       * @param arg_type2 Type of second argument.
       */
      void add(const std::string& name,
	       const expr_type& result_type,
	       const expr_type& arg_type1,
	       const expr_type& arg_type2) {
	std::vector<expr_type> arg_types;
	arg_types.push_back(arg_type1);
	arg_types.push_back(arg_type2);
	add(name,result_type,arg_types);
      }

      /**
       * Add a signature for the ternary function with specified name,
       * result type and argument types.
       *
       * @param name Name of function.
       * @param result_type Return type for function.
       * @param arg_type1 Type of first argument.
       * @param arg_type2 Type of second argument.
       * @param arg_type3 Type of third argument.
       */
      void add(const std::string& name,
	       const expr_type& result_type,
	       const expr_type& arg_type1,
	       const expr_type& arg_type2,
	       const expr_type& arg_type3) {
	std::vector<expr_type> arg_types;
	arg_types.push_back(arg_type1);
	arg_types.push_back(arg_type2);
	arg_types.push_back(arg_type3);
	add(name,result_type,arg_types);
      }

      /**
       * Add a signature for the quaternary function with specified name,
       * result type and argument types.
       *
       * @param name Name of function.
       * @param result_type Return type for function.
       * @param arg_type1 Type of first argument.
       * @param arg_type2 Type of second argument.
       * @param arg_type3 Type of third argument.
       * @param arg_type4 Type of fourth argument.
       */
      void add(const std::string& name,
	       const expr_type& result_type,
	       const expr_type& arg_type1,
	       const expr_type& arg_type2,
	       const expr_type& arg_type3,
	       const expr_type& arg_type4) {
	std::vector<expr_type> arg_types;
	arg_types.push_back(arg_type1);
	arg_types.push_back(arg_type2);
	arg_types.push_back(arg_type3);
	arg_types.push_back(arg_type4);
	add(name,result_type,arg_types);
      }

      void add_unary(const::std::string& name) {
	add(name,DOUBLE_T,DOUBLE_T);
      }

      void add_binary(const::std::string& name) {
	add(name,DOUBLE_T,DOUBLE_T,DOUBLE_T);
      }

      void add_ternary(const::std::string& name) {
	add(name,DOUBLE_T,DOUBLE_T,DOUBLE_T,DOUBLE_T);
      }

      unsigned int num_promotions(const std::vector<expr_type>& call_args,
			 const std::vector<expr_type>& sig_args) {
	int num_promotions = 0U;
	for (unsigned int i = 0; i < call_args.size(); ++i) {
	  if (call_args[i] == sig_args[i]) {
	    continue;
	  } else if (call_args[i].is_primitive_int()
		     && sig_args[i].is_primitive_double()) {
	    ++num_promotions;
	  } else {
	    return -1;
	  } 
	}
	return num_promotions;
      }

      /**
       * Return the result type for the function with the specified
       * name and argument types.  If the function and argument pair
       * do not have a result type defined, return the dmmy
       * expression type.
       *
       * @param name Name of function.
       * @param args Sequence of argument expression types.
       * @return Result type of function with specified name and
       * argument types.
       */
      expr_type get_result_type(const std::string& name,
				const std::vector<expr_type>& args) {
	std::vector<function_signature_t> signatures = sigs_map_[name];
	unsigned int match_index = 0U; 
	unsigned int min_promotions = UINT_MAX; 
	unsigned int num_matches = 0U;

	for (unsigned int i = 0; i < signatures.size(); ++i) {
	  int promotions = num_promotions(args,signatures[i].second);
	  if (promotions < 0) continue; // no match
	  if (promotions < min_promotions) {
	    min_promotions = promotions;
	    match_index = i;
	    num_matches = 1U;
	  } else if (promotions == min_promotions) {
	    ++num_matches;
	  }
	}

	if (num_matches == 1) {
	  return signatures[match_index].first;
	} else if (num_matches == 0U) {
	  std::cerr << "no matches for function name=\"" << name << "\"" 
		    << std::endl;
	} else {
	  std::cerr << num_matches << " matches with " 
		    << min_promotions << " integer promotions "
		    << "for function name=\"" << name << "\"" << std::endl;
	}
	for (unsigned int i = 0; i < args.size(); ++i)
	  std::cerr << "    arg " << i << " type=" << args[i] << std::endl;
	return expr_type(); // ill-formed dummy
      }
      
    private:

      function_signatures() { 
#include <stan/gm/function_signatures.hpp>
      }

      function_signatures(const function_signatures& fs);

      std::map<std::string, std::vector<function_signature_t> > sigs_map_;

      static function_signatures* sigs_;  // init below outside of class
    };

    function_signatures* function_signatures::sigs_ = 0;

    struct distribution {
      distribution() {
      }
      distribution(std::string& family,
		   std::vector<expression>& args)
	: family_(family),
	  args_(args) {
      }
      std::string family_;
      std::vector<expression> args_;
    };

    struct statements {
      statements() {
      }
      statements(std::vector<statement> stmts)
	: statements_(stmts) {
      }
      std::vector<statement> statements_;
    };

    struct expression_type_vis : public boost::static_visitor<expr_type> {
      expr_type operator()(const nil& e) const {
	return expr_type();
      }
      template <typename T>
      expr_type operator()(const T& e) const {
	return e.type_;
      }
    };


    struct expression {
      typedef boost::variant<nil, 
			     boost::recursive_wrapper<int_literal>,
			     boost::recursive_wrapper<double_literal>,
			     boost::recursive_wrapper<variable>,
			     boost::recursive_wrapper<fun>,
                             boost::recursive_wrapper<index_op>,
			     boost::recursive_wrapper<binary_op>,
			     boost::recursive_wrapper<unary_op> > 
      expression_t;

      expr_type expression_type() const {
	expression_type_vis vis;
	return boost::apply_visitor(vis,expr_);
      }

      expression()
	: expr_(nil()) {
      }

      expression(const expression& e) 
	: expr_(e.expr_) {
      }

      template <typename Expr>
      expression(const Expr& expr)
	: expr_(expr) {
      }

      expression& operator+=(const expression& rhs);
      expression& operator-=(const expression& rhs);
      expression& operator*=(const expression& rhs);
      expression& operator/=(const expression& rhs);

      expression_t expr_;
    };

    struct is_nil_op : public boost::static_visitor<bool> {
      bool operator()(nil const& x) const { return true; }
      
      template <typename T>
      bool operator()(const T& x) const { return false; }
    };

    bool is_nil(const expression& e) {
      is_nil_op ino;
      return boost::apply_visitor(ino,e.expr_);
    }

    struct var {
      var() { } // req for FUSION_ADAPT
      var(std::string const& name,
	  std::vector<expression> const& dims) 
	: name_(name),
	  dims_(dims) {
      }
      std::string name_;
      std::vector<expression> dims_;
    };

    struct int_literal {
      int_literal()
	: type_(INT_T) { 
      }
      int_literal(int val) 
      : val_(val), 
	type_(INT_T) { 
      }
      int_literal(const int_literal& il) 
	: val_(il.val_),
	  type_(il.type_) {
      }
      int_literal& operator=(const int_literal& il) {
	val_ = il.val_;
	type_ = il.type_;
	return *this;
      }
      int val_;
      expr_type type_;
    };

    struct double_literal {
      double_literal() 
	: type_(DOUBLE_T,0U) { 
      }
      double_literal& operator=(const double_literal& dl) {
	val_ = dl.val_;
	type_ = dl.type_;
	return *this;
      }
      double_literal(double val)
      : val_(val),
	type_(DOUBLE_T,0U) {
      }
      double val_;
      expr_type type_;
    };

    // struct base_var_decl;

    struct variable {
      variable() { }
      variable(std::string name) : name_(name) { }
      void set_type(const base_expr_type& base_type, 
		    unsigned int num_dims) {
	type_ = expr_type(base_type, num_dims);
      }
      std::string name_;
      expr_type type_;
    };

    
    // add_function_signature("normal",
    // expr_type(DOUBLE_T,0));
			   

    struct fun {
      fun() { }
      fun(std::string const& name,
	  std::vector<expression> const& args) 
	: name_(name),
	  args_(args),
	  type_() { // FIXME: plumb in typer
      }
      std::string name_;
      std::vector<expression> args_;
      expr_type type_;
    };
    
    unsigned int total_dims(const std::vector<std::vector<expression> >& dimss) {
      unsigned int total = 0U;
      for (unsigned int i = 0; i < dimss.size(); ++i)
	total += dimss[i].size();
      return total;
    }

    struct index_op {
      index_op() { }
      // vec of vec for e.g., e[1,2][3][4,5,6]
      index_op(const expression& expr,
	       const std::vector<std::vector<expression> >& dimss) 
	: expr_(expr),
	  dimss_(dimss),
	  type_(expr.expression_type().type(),
		expr.expression_type().num_dims() - total_dims(dimss)) {
      }
      expression expr_;
      std::vector<std::vector<expression> > dimss_;
      expr_type type_;
    };

    struct binary_op {
      binary_op() {
      }
      binary_op(const expression& left,
		char op,
		const expression& right)
        : op(op), 
	  left(left), 
	  right(right),
	  type_(promote_primitive(left.expression_type(),
				  right.expression_type())) {
      }
      char op;
      expression left;
      expression right;
      expr_type type_;
    };

    struct unary_op {
      unary_op(char op,
	       expression const& subject)
        : op(op), 
	  subject(subject),
	  type_(promote_primitive(subject.expression_type())) {
      }
      char op;
      expression subject;
      expr_type type_;
    };

    struct range {
      range() { } 
      range(expression const& low,
	    expression const& high)
	: low_(low),
	  high_(high) {
      }
      expression low_;
      expression high_;
    };

    struct base_var_decl {
      std::string name_;
      std::vector<expression> dims_;
      base_expr_type base_type_;
      base_var_decl() { }
      base_var_decl(const base_expr_type& base_type) 
      : base_type_(base_type) {
      }
      base_var_decl(const std::string& name,
		    const std::vector<expression>& dims,
		    const base_expr_type& base_type)
      : name_(name),
	dims_(dims),
	base_type_(base_type) {
      }
    };


    struct int_var_decl : public base_var_decl {
      int_var_decl() : base_var_decl(INT_T) { }
      int_var_decl(range const& range,
		   std::string const& name,
		   std::vector<expression> const& dims) 
	: base_var_decl(name,dims,INT_T),
	  range_(range) {
      }
      range range_;
    };

    struct double_var_decl : public base_var_decl {
      double_var_decl() : base_var_decl(DOUBLE_T) { }
      double_var_decl(range const& range,
		      std::string const& name,
		      std::vector<expression> const& dims)
	: base_var_decl(name,dims,DOUBLE_T),
	  range_(range) {
      }
      range range_;
    };

    struct simplex_var_decl : public base_var_decl {
      simplex_var_decl() : base_var_decl(VECTOR_T) { }
      simplex_var_decl(expression const& K,
		       std::string const& name,
		       std::vector<expression> const& dims)
	: base_var_decl(name,dims,VECTOR_T),
	  K_(K) {
      }
      expression K_;
    };

    struct pos_ordered_var_decl : public base_var_decl {
      pos_ordered_var_decl() : base_var_decl(VECTOR_T) { }
      pos_ordered_var_decl(expression const& K,
			   std::string const& name,
			   std::vector<expression> const& dims)
	: base_var_decl(name,dims,VECTOR_T),
	  K_(K) {
      }
      std::string name_;
      expression K_;
      std::vector<expression> dims_;
    };

    struct vector_var_decl : public base_var_decl {
      vector_var_decl() : base_var_decl(VECTOR_T) { }
      vector_var_decl(expression const& M,
		      std::string const& name,
		      std::vector<expression> const& dims)
	: base_var_decl(name,dims,VECTOR_T),
	  M_(M) {
      }
      expression M_;
    };

    struct row_vector_var_decl : public base_var_decl {
      row_vector_var_decl() : base_var_decl(ROW_VECTOR_T) { }
      row_vector_var_decl(expression const& N,
			  std::string const& name,
			  std::vector<expression> const& dims)
	: base_var_decl(name,dims,ROW_VECTOR_T),
	  N_(N) {
      }
      expression N_;
    };

    struct matrix_var_decl : public base_var_decl {
      matrix_var_decl() : base_var_decl(MATRIX_T) { }
      matrix_var_decl(expression const& M,
		      expression const& N,
		      std::string const& name,
		      std::vector<expression> const& dims)
	: base_var_decl(name,dims,MATRIX_T),
	  M_(M),
	  N_(N) {
      }
      expression M_;
      expression N_;
    };

    struct cov_matrix_var_decl : public base_var_decl {
      cov_matrix_var_decl() : base_var_decl(MATRIX_T) { }
      cov_matrix_var_decl(expression const& K,
			   std::string const& name,
			   std::vector<expression> const& dims)
	: base_var_decl(name,dims,MATRIX_T),
	  K_(K) {
      }
      std::string name_;
      expression K_;
      std::vector<expression> dims_;
    };

    struct corr_matrix_var_decl : public base_var_decl {
      corr_matrix_var_decl() : base_var_decl(MATRIX_T) { }
      corr_matrix_var_decl(expression const& K,
			   std::string const& name,
			   std::vector<expression> const& dims)
	: base_var_decl(name,dims,MATRIX_T),
	  K_(K) {
      }
      std::string name_;
      expression K_;
      std::vector<expression> dims_;
    };

    struct var_decl {
      typedef boost::variant<nil, // just for default constructor
			     boost::recursive_wrapper<int_var_decl>,
			     boost::recursive_wrapper<double_var_decl>,
			     boost::recursive_wrapper<vector_var_decl>,
			     boost::recursive_wrapper<row_vector_var_decl>,
			     boost::recursive_wrapper<matrix_var_decl>,
			     boost::recursive_wrapper<simplex_var_decl>,
			     boost::recursive_wrapper<pos_ordered_var_decl>,
			     boost::recursive_wrapper<cov_matrix_var_decl>,
			     boost::recursive_wrapper<corr_matrix_var_decl> >
      type;

      var_decl() : decl_(nil()) { }

      template <typename Decl>
      var_decl(Decl const& decl) : decl_(decl) { }

      type decl_;
    };

    struct statement {
      typedef boost::variant<nil,
			     boost::recursive_wrapper<assignment>,
			     boost::recursive_wrapper<sample>,
			     boost::recursive_wrapper<statements>,
			     boost::recursive_wrapper<for_statement> >
      type;

      statement()
	: statement_(nil()) {
      }

      template <typename Statement>
      statement(Statement const& statement)
	: statement_(statement) {
      }

      type statement_;
    };

    struct for_statement {
      std::string variable_;
      range range_;
      statement statement_;
      for_statement() {
      }
      for_statement(std::string& variable,
		    range& range,
		    statement& stmt)
	: variable_(variable),
	  range_(range),
	  statement_(stmt) {
      }
    };

    struct program {
      program() { }
      program(const std::vector<var_decl> & data_decl,
	      const std::pair<std::vector<var_decl>,std::vector<statement> >& derived_data_decl,
	      const std::vector<var_decl>& parameter_decl,
	      const std::pair<std::vector<var_decl>,std::vector<statement> >& derived_decl,
	      statement const& stmt) 
	: data_decl_(data_decl),
	  derived_data_decl_(derived_data_decl),
	  parameter_decl_(parameter_decl),
	  derived_decl_(derived_decl),
	  statement_(stmt) {
      }
      std::vector<var_decl> data_decl_;
      std::pair<std::vector<var_decl>,std::vector<statement> > derived_data_decl_;
      std::vector<var_decl> parameter_decl_;
      std::pair<std::vector<var_decl>,std::vector<statement> > derived_decl_;
      statement statement_;
    };

    struct sample {
      sample() {
      }
      sample(expression& e,
	     distribution& dist) 
	: expr_(e),
	  dist_(dist) {
      }
      expression expr_;
      distribution dist_;
    };

    struct assignment {
      assignment() {
      }
      assignment(var& var,
		 expression& expr)
	: var_(var),
	  expr_(expr) {
      }
      var var_;
      expression expr_;
    };
      
    void generate_expression(const expression& e, std::ostream& o);

    expression& expression::operator+=(const expression& rhs) {
      expr_ = binary_op(expr_, '+', rhs);
      return *this;
    }

    expression& expression::operator-=(const expression& rhs) {
      expr_ = binary_op(expr_, '-', rhs);
      return *this;
    }

    expression& expression::operator*=(expression const& rhs) {
      expr_ = binary_op(expr_, '*', rhs);
      return *this;
    }

    expression& expression::operator/=(expression const& rhs) {
      expr_ = binary_op(expr_, '/', rhs);
      return *this;
    }

  }
}

#endif
