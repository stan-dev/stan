#ifndef STAN__GM__PARSER__VAR_DECLS_GRAMMAR_DEF__HPP
#define STAN__GM__PARSER__VAR_DECLS_GRAMMAR_DEF__HPP

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
#include <boost/spirit/include/qi_numeric.hpp>
#include <stan/gm/ast.hpp>
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
                          (stan::gm::range, range_)
                          (stan::gm::expression, M_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::row_vector_var_decl,
                          (stan::gm::range, range_)
                          (stan::gm::expression, N_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::matrix_var_decl,
                          (stan::gm::range, range_)
                          (stan::gm::expression, M_)
                          (stan::gm::expression, N_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::unit_vector_var_decl,
                          (stan::gm::expression, K_)
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

BOOST_FUSION_ADAPT_STRUCT(stan::gm::positive_ordered_var_decl,
                          (stan::gm::expression, K_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::cholesky_factor_var_decl,
                          (stan::gm::expression, M_)
                          (stan::gm::expression, N_)
                          (std::string, name_)
                          (std::vector<stan::gm::expression>, dims_) )

BOOST_FUSION_ADAPT_STRUCT(stan::gm::cholesky_corr_var_decl,
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
      bool operator()(const nil& /*x*/) const { 
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
      bool operator()(const vector_var_decl& /*x*/) const {
        return true;
      }
      bool operator()(const row_vector_var_decl& /*x*/) const {
        return true;
      }
      bool operator()(const matrix_var_decl& /*x*/) const {
        return true;
      }
      bool operator()(const unit_vector_var_decl& /*x*/) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found unit_vector." << std::endl;
        return false;
      }
      bool operator()(const simplex_var_decl& /*x*/) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found simplex." << std::endl;
        return false;
      }
      bool operator()(const ordered_var_decl& /*x*/) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found ordered." << std::endl;
        return false;
      }
      bool operator()(const positive_ordered_var_decl& /*x*/) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found positive_ordered." << std::endl;
        return false;
      }
      bool operator()(const cholesky_factor_var_decl& /*x*/) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found cholesky_factor." << std::endl;
        return false;
      }
      bool operator()(const cholesky_corr_var_decl& /*x*/) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found cholesky_factor_corr." << std::endl;
        return false;
      }
      bool operator()(const cov_matrix_var_decl& /*x*/) const {
        error_msgs_ << "require unconstrained variable declaration."
                    << " found cov_matrix." << std::endl;
        return false;
      }
      bool operator()(const corr_matrix_var_decl& /*x*/) const {
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
      bool operator()(const nil& /*e*/) const {
        return true;
      }
      bool operator()(const int_literal& /*x*/) const {
        return true;
      }
      bool operator()(const double_literal& /*x*/) const {
        return true;
      }
      bool operator()(const array_literal& x) const {
        for (size_t i = 0; i < x.args_.size(); ++i)
          if (!boost::apply_visitor(*this,x.args_[i].expr_))
            return false;
        return true;
      }
      bool operator()(const variable& x) const {
        var_origin origin = var_map_.get_origin(x.name_);
        bool is_data = (origin == data_origin) 
          || (origin == transformed_data_origin)
          || (origin == local_origin);
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
      bool operator()(const integrate_ode& x) const {
        return boost::apply_visitor(*this, x.y0_.expr_)
          && boost::apply_visitor(*this, x.theta_.expr_);
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
      template <typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
      struct result { typedef void type; };
      // each type derived from base_var_decl gets own instance
      template <typename R, typename T>
      void operator()(R& var_decl_result,
                      const T& var_decl, 
                      variable_map& vm,
                      bool& pass,
                      const var_origin& vo,
                      std::ostream& error_msgs) const {
        if (vm.exists(var_decl.name_)) {
          // variable already exists
          pass = false;
          error_msgs << "duplicate declaration of variable, name="
                     << var_decl.name_;

          error_msgs << "; attempt to redeclare as ";
          print_var_origin(error_msgs,vo);  // FIXME -- need original vo

          error_msgs << "; original declaration as ";
          print_var_origin(error_msgs,vm.get_origin(var_decl.name_));

          error_msgs << std::endl;
          var_decl_result = var_decl;
          return;
        } 
        if ((vo == parameter_origin || vo == transformed_parameter_origin)
            && var_decl.base_type_ == INT_T) {
          pass = false;
          error_msgs << "integer parameters or transformed parameters are not allowed; "
                     << " found declared type int, parameter name=" << var_decl.name_
                     << std::endl;
          var_decl_result = var_decl;
          return;
        }
        pass = true;  // probably don't need to set true
        vm.add(var_decl.name_,var_decl,vo);
        var_decl_result = var_decl;
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
      std::set<std::string> reserved_word_set_;
      std::set<std::string> const_fun_name_set_;

      template <typename T1, typename T2>
      struct result { typedef bool type; };

      void reserve(const std::string& w) {
        reserved_word_set_.insert(w);
      }

      template <typename S, typename T>
      static bool contains(const S& s,
                           const T& x) {
        return s.find(x) != s.end();
      }

      bool identifier_exists(const std::string& identifier) const {
        return contains(reserved_word_set_, identifier)
          || ( contains(function_signatures::instance().key_set(), identifier)
               && !contains(const_fun_name_set_, identifier) );
      }

      validate_identifier() {
        // Constant functions which can be used as identifiers
        const_fun_name_set_.insert("pi");
        const_fun_name_set_.insert("e");
        const_fun_name_set_.insert("sqrt2");
        const_fun_name_set_.insert("log2");
        const_fun_name_set_.insert("log10");
        const_fun_name_set_.insert("not_a_number");
        const_fun_name_set_.insert("positive_infinity");
        const_fun_name_set_.insert("negative_infinity");
        const_fun_name_set_.insert("epsilon");
        const_fun_name_set_.insert("negative_epsilon");

        // illegal identifiers
        reserve("for");  
        reserve("in");  
        reserve("while");
        reserve("repeat");  
        reserve("until");  
        reserve("if");
        reserve("then"); 
        reserve("else"); 
        reserve("true");  
        reserve("false");

        reserve("int");
        reserve("real"); 
        reserve("vector"); 
        reserve("unit_vector");
        reserve("simplex"); 
        reserve("ordered"); 
        reserve("positive_ordered"); 
        reserve("row_vector"); 
        reserve("matrix"); 
        reserve("cholesky_factor_cov");
        reserve("cholesky_factor_corr");
        reserve("cov_matrix");
        reserve("corr_matrix"); 

        
        reserve("model"); 
        reserve("data"); 
        reserve("parameters"); 
        reserve("quantities"); 
        reserve("transformed"); 
        reserve("generated");
        
        reserve("var");
        
        reserve("alignas"); 
        reserve("alignof"); 
        reserve("and"); 
        reserve("and_eq"); 
        reserve("asm"); 
        reserve("auto"); 
        reserve("bitand"); 
        reserve("bitor"); 
        reserve("bool"); 
        reserve("break"); 
        reserve("case"); 
        reserve("catch"); 
        reserve("char"); 
        reserve("char16_t"); 
        reserve("char32_t"); 
        reserve("class"); 
        reserve("compl"); 
        reserve("const"); 
        reserve("constexpr"); 
        reserve("const_cast"); 
        reserve("continue"); 
        reserve("decltype"); 
        reserve("default"); 
        reserve("delete"); 
        reserve("do"); 
        reserve("double"); 
        reserve("dynamic_cast"); 
        reserve("else"); 
        reserve("enum"); 
        reserve("explicit"); 
        reserve("export"); 
        reserve("extern"); 
        reserve("false"); 
        reserve("float"); 
        reserve("for"); 
        reserve("friend"); 
        reserve("goto"); 
        reserve("if"); 
        reserve("inline"); 
        reserve("int"); 
        reserve("long"); 
        reserve("mutable"); 
        reserve("namespace"); 
        reserve("new"); 
        reserve("noexcept"); 
        reserve("not"); 
        reserve("not_eq"); 
        reserve("nullptr"); 
        reserve("operator"); 
        reserve("or"); 
        reserve("or_eq"); 
        reserve("private"); 
        reserve("protected"); 
        reserve("public"); 
        reserve("register"); 
        reserve("reinterpret_cast"); 
        reserve("return"); 
        reserve("short"); 
        reserve("signed"); 
        reserve("sizeof"); 
        reserve("static"); 
        reserve("static_assert"); 
        reserve("static_cast"); 
        reserve("struct"); 
        reserve("switch"); 
        reserve("template"); 
        reserve("this"); 
        reserve("thread_local"); 
        reserve("throw"); 
        reserve("true"); 
        reserve("try"); 
        reserve("typedef"); 
        reserve("typeid"); 
        reserve("typename"); 
        reserve("union"); 
        reserve("unsigned"); 
        reserve("using"); 
        reserve("virtual"); 
        reserve("void"); 
        reserve("volatile"); 
        reserve("wchar_t"); 
        reserve("while"); 
        reserve("xor"); 
        reserve("xor_eq");

        // function names declared in signatures
        using stan::gm::function_signatures;
        using std::set;
        using std::string;
        const function_signatures& sigs = function_signatures::instance();

        set<string> fun_names = sigs.key_set();
        for (set<string>::iterator it = fun_names.begin();  it != fun_names.end();  ++it)
          if (!contains(const_fun_name_set_, *it))
            reserve(*it);
      }

      bool operator()(const std::string& identifier,
                      std::stringstream& error_msgs) const {
        int len = identifier.size();
        if (len >= 2
            && identifier[len-1] == '_'
            && identifier[len-2] == '_') {
          error_msgs << "variable identifier (name) may not end in double underscore (__)"
                     << std::endl
                     << "    found identifer=" << identifier << std::endl;
          return false;
        }
        size_t period_position = identifier.find('.');
        if (period_position != std::string::npos) {
          error_msgs << "variable identifier may not contain a period (.)"
                     << std::endl
                     << "    found period at position (indexed from 0)=" << period_position
                     << std::endl
                     << "    found identifier=" << identifier 
                     << std::endl;
          return false;
        }
        if (identifier_exists(identifier)) { 
          error_msgs << "variable identifier (name) may not be reserved word"
                     << std::endl
                     << "    found identifier=" << identifier 
                     << std::endl;
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_identifier> validate_identifier_f;

    // copies single dimension from M to N if only M declared
    struct copy_square_cholesky_dimension_if_necessary {
      template <typename T1>
      struct result { typedef void type; };
      void operator()(cholesky_factor_var_decl& var_decl) const {
        if (is_nil(var_decl.N_))
          var_decl.N_ = var_decl.M_;
      }
    };
    boost::phoenix::function<copy_square_cholesky_dimension_if_necessary>
    copy_square_cholesky_dimension_if_necessary_f;

    struct empty_range {
      template <typename T1>
      struct result { typedef range type; };
      range operator()(std::stringstream& /*error_msgs*/) const {
        return range();
      }
    };
    boost::phoenix::function<empty_range> empty_range_f;

    struct validate_int_expr {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };

      void operator()(const expression& expr,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        if (!expr.expression_type().is_primitive_int()) {
          error_msgs << "expression denoting integer required; found type=" 
                     << expr.expression_type() << std::endl;

          pass = false;
          return;
        }
        pass = true;
      }
    };
    boost::phoenix::function<validate_int_expr> validate_int_expr_f;

    struct set_int_range_lower {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };
      void operator()(range& range,
                      const expression& expr,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        range.low_ = expr;
        validate_int_expr validator;
        validator(expr,pass,error_msgs);
      }
    };
    boost::phoenix::function<set_int_range_lower> set_int_range_lower_f;

    struct set_int_range_upper {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };
      void operator()(range& range,
                      const expression& expr,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        range.high_ = expr;
        validate_int_expr validator;
        validator(expr,pass,error_msgs);
      }
    };
    boost::phoenix::function<set_int_range_upper> set_int_range_upper_f;



    struct validate_int_data_expr {
      template <typename T1, typename T2, typename T3, typename T4, typename T5>
      struct result { typedef void type; };

      void operator()(const expression& expr,
                      int var_origin,
                      bool& pass,
                      variable_map& var_map,
                      std::stringstream& error_msgs) const {
        if (!expr.expression_type().is_primitive_int()) {
          error_msgs << "dimension declaration requires expression denoting integer;"
                     << " found type=" 
                     << expr.expression_type() 
                     << std::endl;
          pass = false;
        } else if (var_origin != local_origin) {
          data_only_expression vis(error_msgs,var_map);
          bool only_data_dimensions = boost::apply_visitor(vis,expr.expr_);
          pass = only_data_dimensions;
        } else {
          // don't need to check data vs. parameter in dimensions for
          // local variable declarations
          pass = true;
        }
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
          error_msgs << "expression denoting real required; found type=" 
                     << expr.expression_type() << std::endl;
          return false;
        }
        return true;
      }
    };
    boost::phoenix::function<validate_double_expr> validate_double_expr_f;


    struct set_double_range_lower {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };
      void operator()(range& range,
                      const expression& expr,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        range.low_ = expr;
        validate_double_expr validator;
        pass = validator(expr,error_msgs);
      }
    };
    boost::phoenix::function<set_double_range_lower> set_double_range_lower_f;

    struct set_double_range_upper {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };
      void operator()(range& range,
                      const expression& expr,
                      bool& pass,
                      std::stringstream& error_msgs) const {
        range.high_ = expr;
        validate_double_expr validator;
        pass = validator(expr,error_msgs);
      }
    };
    boost::phoenix::function<set_double_range_upper> set_double_range_upper_f;


    template <typename Iterator>
    var_decls_grammar<Iterator>::var_decls_grammar(variable_map& var_map,
                                                   std::stringstream& error_msgs)
      : var_decls_grammar::base_type(var_decls_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        // expression_g allows full recursion
        expression_g(var_map,error_msgs),
        // expression07_g disallows comparisons
        expression07_g(var_map,error_msgs,expression_g)
    {

      using boost::spirit::qi::_1;
      using boost::spirit::qi::_3;
      using boost::spirit::qi::char_;
      using boost::spirit::qi::eps;
      using boost::spirit::qi::lexeme;
      using boost::spirit::qi::lit;
      using boost::spirit::qi::no_skip;
      using boost::spirit::qi::_pass;
      using boost::spirit::qi::_val;
      using boost::spirit::qi::labels::_a;
      using boost::spirit::qi::labels::_r1;
      using boost::spirit::qi::labels::_r2;

      var_decls_r.name("variable declarations");
      var_decls_r 
        %= *(var_decl_r(_r1,_r2)) 
        ;

      // _a = error state local, 
      // _r1 constraints allowed inherited,
      // _r2 var_origin
      var_decl_r.name("variable declaration");
      var_decl_r 
        %= (int_decl_r(_r2)             
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs))]
            | double_decl_r(_r2)        
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs_))]
            | vector_decl_r(_r2)        
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs_))]
            | row_vector_decl_r(_r2)    
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs_))]
            | matrix_decl_r(_r2)
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs_))]
            | unit_vector_decl_r(_r2)       
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs_))]
            | simplex_decl_r(_r2)
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs_))]
            | ordered_decl_r(_r2)
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs_))]
            | positive_ordered_decl_r(_r2)   
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs_))]
            | cholesky_factor_decl_r(_r2)    
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs_))]
            | cholesky_corr_decl_r(_r2)    
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                              boost::phoenix::ref(error_msgs_))]
            | cov_matrix_decl_r(_r2)    
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs_))]
            | corr_matrix_decl_r(_r2)   
            [add_var_f(_val,_1,boost::phoenix::ref(var_map_),_a,_r2,
                       boost::phoenix::ref(error_msgs_))]
            )
        > lit(';')
        [_pass 
         = validate_decl_constraints_f(_r1,_a,_val,
                                       boost::phoenix::ref(error_msgs_))]
        ;

      int_decl_r.name("integer declaration");
      int_decl_r 
        %= ( lit("int")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > -range_brackets_int_r(_r1)
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      double_decl_r.name("real declaration");
      double_decl_r 
        %= ( lit("real")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > -range_brackets_double_r(_r1)
        > identifier_r
        > opt_dims_r(_r1)
        ;

      vector_decl_r.name("vector declaration");
      vector_decl_r 
        %= ( lit("vector")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > -range_brackets_double_r(_r1)
        > lit('[')
        > expression_g(_r1)
        [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(']')
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      row_vector_decl_r.name("row vector declaration");
      row_vector_decl_r 
        %= ( lit("row_vector")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > -range_brackets_double_r(_r1)
        > lit('[')
        > expression_g(_r1)
        [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(']')
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      matrix_decl_r.name("matrix declaration");
      matrix_decl_r 
        %= ( lit("matrix")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > -range_brackets_double_r(_r1)
        > lit('[')
        > expression_g(_r1)
          [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(',')
        > expression_g(_r1)
          [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(']')
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      unit_vector_decl_r.name("unit_vector declaration");
      unit_vector_decl_r 
        %= ( lit("unit_vector")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > lit('[')
        > expression_g(_r1)
          [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(']')
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      simplex_decl_r.name("simplex declaration");
      simplex_decl_r 
        %= ( lit("simplex")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > lit('[')
        > expression_g(_r1)
          [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(']')
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      ordered_decl_r.name("ordered declaration");
      ordered_decl_r 
        %= ( lit("ordered")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > lit('[')
        > expression_g(_r1)
          [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(']')
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      positive_ordered_decl_r.name("positive_ordered declaration");
      positive_ordered_decl_r 
        %= ( lit("positive_ordered")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > lit('[')
        > expression_g(_r1)
          [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(']')
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      cholesky_factor_decl_r.name("cholesky factor for symmetric, positive-def declaration");
      cholesky_factor_decl_r 
        %= ( lit("cholesky_factor_cov") 
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > lit('[')
        > expression_g(_r1)
          [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > -( lit(',')
             > expression_g(_r1)
               [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
             ) 
        > lit(']') 
        > identifier_r 
        > opt_dims_r(_r1)
        > eps
        [copy_square_cholesky_dimension_if_necessary_f(_val)]
        ;

      cholesky_corr_decl_r.name("cholesky factor for correlation matrix declaration");
      cholesky_corr_decl_r 
        %= ( lit("cholesky_factor_corr")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > lit('[')
        > expression_g(_r1)
          [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(']') 
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      cov_matrix_decl_r.name("covariance matrix declaration");
      cov_matrix_decl_r 
        %= ( lit("cov_matrix")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > lit('[')
        > expression_g(_r1)
          [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(']')
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      corr_matrix_decl_r.name("correlation matrix declaration");
      corr_matrix_decl_r 
        %= ( lit("corr_matrix")
             >> no_skip[!char_("a-zA-Z0-9_")] )
        > lit('[')
        > expression_g(_r1)
          [validate_int_expr_f(_1,_pass,boost::phoenix::ref(error_msgs_))]
        > lit(']')
        > identifier_r 
        > opt_dims_r(_r1)
        ;

      opt_dims_r.name("array dimensions (optional)");
      opt_dims_r 
        %=  - dims_r(_r1);

      dims_r.name("array dimensions");
      dims_r 
        %= lit('[') 
        > (expression_g(_r1)
           [validate_int_data_expr_f(_1,_r1,_pass,
                                     boost::phoenix::ref(var_map_),
                                     boost::phoenix::ref(error_msgs_))]
           % ',')
        > lit(']')
        ;

      range_brackets_int_r.name("integer range expression pair, brackets");
      range_brackets_int_r 
        = lit('<') [_val = empty_range_f(boost::phoenix::ref(error_msgs_))]
        >> ( 
           ( (lit("lower")
              >> lit('=')
              >> expression07_g(_r1)
                 [set_int_range_lower_f(_val,_1,_pass,
                                        boost::phoenix::ref(error_msgs_)) ])
             >> -( lit(',')
                   >> lit("upper")
                   >> lit('=')
                   >> expression07_g(_r1)
                      [set_int_range_upper_f(_val,_1,_pass,
                                             boost::phoenix::ref(error_msgs_)) ] ) )
           | 
           ( lit("upper")
             >> lit('=')
             >> expression07_g(_r1)
                [set_int_range_upper_f(_val,_1,_pass,
                                       boost::phoenix::ref(error_msgs_)) ])
            )
        >> lit('>');

      range_brackets_double_r.name("real range expression pair, brackets");
      range_brackets_double_r 
        = lit('<') [_val = empty_range_f(boost::phoenix::ref(error_msgs_))]
        > ( 
           ( (lit("lower")
              > lit('=')
              > expression07_g(_r1)
                [set_double_range_lower_f(_val,_1,_pass,
                                          boost::phoenix::ref(error_msgs_)) ])
             > -( lit(',')
                  > lit("upper")
                  > lit('=')
                  > expression07_g(_r1)
                  [set_double_range_upper_f(_val,_1,_pass,
                                            boost::phoenix::ref(error_msgs_)) ] ) )
           | 
           ( lit("upper")
             > lit('=')
             > expression07_g(_r1)
               [set_double_range_upper_f(_val,_1,_pass,
                                         boost::phoenix::ref(error_msgs_)) ])
            )
        > lit('>');

      identifier_r.name("identifier");
      identifier_r
        %= identifier_name_r
           [_pass = validate_identifier_f(_val,boost::phoenix::ref(error_msgs_))]
        ;

      identifier_name_r.name("identifier subrule");
      identifier_name_r
        %= lexeme[char_("a-zA-Z") 
                  >> *char_("a-zA-Z0-9_.")]
        ;
        
    }
  }

  
}
#endif
