#ifndef STAN__GM__PARSER__FUNCTIONS__GRAMMAR_DEF__HPP__
#define STAN__GM__PARSER__FUNCTIONS__GRAMMAR_DEF__HPP__

#include <set>
#include <utility>
#include <vector>

#include <boost/spirit/include/qi.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
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

#include <boost/spirit/include/version.hpp>
#include <boost/spirit/include/support_line_pos_iterator.hpp>

#include <stan/gm/ast.hpp>
#include <stan/gm/grammars/functions_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>


BOOST_FUSION_ADAPT_STRUCT(stan::gm::function_decl_def,
                          (stan::gm::expr_type, return_type_)
                          (std::string, name_) 
                          (std::vector<stan::gm::arg_decl>, arg_decls_)
                          (stan::gm::statement, body_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::arg_decl,
                          (stan::gm::expr_type, arg_type_)
                          (std::string, name_)
                          (stan::gm::statement, body_) );

namespace stan {

  namespace gm {

    struct validate_non_void_arg_function {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };
      void operator()(const expr_type& arg_type,
                      bool& pass,
                      std::ostream& error_msgs) const {
        pass = !arg_type.is_void();
        if (!pass)
          error_msgs << "Functions cannot contain void argument types; "
                     << "found void argument."
                     << std::endl;
      }
    };
    boost::phoenix::function<validate_non_void_arg_function> validate_non_void_arg_f;

    struct set_void_function {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };
      void operator()(const expr_type& return_type,
                      var_origin& origin,
                      bool& pass,
                      std::ostream& error_msgs) const {
        if (return_type.is_void() && return_type.num_dims() > 0) {
          error_msgs << "Void return type may not have dimensions declared."
                     << std::endl;
          pass = false;
          return;
        }
        origin = return_type.is_void()
          ? void_function_argument_origin 
          : function_argument_origin;
      }
    };
    boost::phoenix::function<set_void_function> set_void_function_f;

    struct set_allows_sampling_origin {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };
      void operator()(const std::string& identifier,
                      bool& allow_sampling,
                      int& origin) const {
        bool is_void_function_origin = (origin == void_function_argument_origin);
        if (ends_with("_lp", identifier)) {
          allow_sampling = true;
          origin = is_void_function_origin
            ? void_function_argument_origin_lp
            : function_argument_origin_lp;
        } else if (ends_with("_rng", identifier)) {
          allow_sampling = false;
          origin = is_void_function_origin
            ? void_function_argument_origin_rng
            : function_argument_origin_rng;
        } else {
          allow_sampling = false;
          origin = is_void_function_origin
            ? void_function_argument_origin
            : function_argument_origin;
        }
      }
    };
    boost::phoenix::function<set_allows_sampling_origin> set_allows_sampling_origin_f;

    struct validate_declarations {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };
      void operator()(bool& pass,
                      std::set<std::pair<std::string, 
                                         function_signature_t> >& declared,
                      std::set<std::pair<std::string, 
                                         function_signature_t> >& defined,
                      std::ostream& error_msgs) const {
        using std::set;
        using std::string;
        using std::pair;
        typedef set<pair<string, function_signature_t> >::iterator iterator_t;
        for (iterator_t it = declared.begin(); it != declared.end(); ++it) {
          if (defined.find(*it) == defined.end()) {
            error_msgs <<"Function declared, but not defined."
                       << " Function name=" << (*it).first
                       << std::endl;
            pass = false;
            return;
          }
        }
        pass = true;
      }
    };
    boost::phoenix::function<validate_declarations> validate_declarations_f;

    struct add_function_signature {
      template <typename T1, typename T2, typename T3, typename T4, typename T5>
      struct result { typedef void type; };
      static bool fun_exists(const std::set<std::pair<std::string, 
                                                      function_signature_t> >& existing,
                             const std::pair<std::string,function_signature_t>& name_sig) {
        for (std::set<std::pair<std::string, function_signature_t> >::const_iterator it 
               = existing.begin();
             it != existing.end();
             ++it)
          if (name_sig.first == (*it).first 
              && name_sig.second.second == (*it).second.second)
            return true;  // name and arg sequences match
        return false;
      }
      void operator()(const function_decl_def& decl,
                      bool& pass,
                      std::set<std::pair<std::string, 
                                         function_signature_t> >& functions_declared,
                      std::set<std::pair<std::string, 
                                         function_signature_t> >& functions_defined,
                      std::ostream& error_msgs) const {

        // build up representations
        expr_type result_type(decl.return_type_.base_type_,
                              decl.return_type_.num_dims_);
        std::vector<expr_type> arg_types;
        for (size_t i = 0; i < decl.arg_decls_.size(); ++i)
          arg_types.push_back(expr_type(decl.arg_decls_[i].arg_type_.base_type_,
                                        decl.arg_decls_[i].arg_type_.num_dims_));
        function_signature_t sig(result_type, arg_types);
        std::pair<std::string, function_signature_t> name_sig(decl.name_, sig);
        
        // check that not already declared if just declaration
        if (decl.body_.is_no_op_statement()
            && fun_exists(functions_declared,name_sig)) {
          error_msgs << "Parse Error.  Function already declared, name=" << decl.name_;
          pass = false;
          return;
        }

        // check not already user defined
        if (fun_exists(functions_defined, name_sig)) {
          error_msgs << "Parse Error.  Function already defined, name=" << decl.name_;
          pass = false;
          return;
        }

        // check not already system defined
        if (!fun_exists(functions_declared,name_sig)
            && function_signatures::instance().is_defined(decl.name_, sig)) {
          error_msgs << "Parse Error.  Function system defined, name=" << decl.name_;
          pass = false;
          return;
        }

        // add declaration in local sets and in parser function sigs
        if (functions_declared.find(name_sig) == functions_declared.end()) {
            functions_declared.insert(name_sig);
            function_signatures::instance()
              .add(decl.name_,
                   result_type,arg_types);
            function_signatures::instance()
              .set_user_defined(name_sig);
        }
        
        // add as definition if there's a body
        if (!decl.body_.is_no_op_statement())
          functions_defined.insert(name_sig);
        pass = true;
      }
    };
    boost::phoenix::function<add_function_signature> add_function_signature_f;

    struct validate_return_type {
      template <typename T1, typename T2, typename T3>
      struct result { typedef void type; };
      void operator()(function_decl_def& decl,
                      bool& pass,
                      std::ostream& error_msgs) const {
        pass = decl.body_.is_no_op_statement()
          || stan::gm::returns_type(decl.return_type_, decl.body_, 
                                    error_msgs);
        if (!pass) {
          error_msgs << "Improper return in body of function.";
          return;
        }

        if (ends_with("_log",decl.name_)
            && !decl.return_type_.is_primitive_double()) {
            pass = false;
            error_msgs << "Require real return type for functions ending in _log.";
        }
      }
    };
    boost::phoenix::function<validate_return_type> validate_return_type_f;

    struct scope_lp {
      template <typename T1>
      struct result { typedef void type; };
      void operator()(variable_map& vm) const {
        vm.add("lp__", DOUBLE_T, local_origin);
      }
    };
    boost::phoenix::function<scope_lp> scope_lp_f;
    

    struct unscope_variables {
      template <typename T1, typename T2>
      struct result { typedef void type; };
      void operator()(function_decl_def& decl,
                      variable_map& vm) const {
        vm.remove("lp__");
        for (size_t i = 0; i < decl.arg_decls_.size(); ++i)
          vm.remove(decl.arg_decls_[i].name_);
      }
    };
    boost::phoenix::function<unscope_variables> unscope_variables_f;


    struct add_fun_var {
      template <typename T1, typename T2, typename T3, typename T4>
      struct result { typedef void type; };
      // each type derived from base_var_decl gets own instance
      void operator()(arg_decl& decl,
                      bool& pass,
                      variable_map& vm,
                      std::ostream& error_msgs) const {
        if (vm.exists(decl.name_)) {
          // variable already exists
          pass = false;
          error_msgs << "duplicate declaration of variable, name="
                     << decl.name_;

          error_msgs << "; attempt to redeclare as function argument";

          error_msgs << "; original declaration as ";
          print_var_origin(error_msgs,vm.get_origin(decl.name_));

          error_msgs << std::endl;
          return;
        }

        pass = true;
        vm.add(decl.name_,
               decl.base_variable_declaration(),
               function_argument_origin);
      }
    };
    boost::phoenix::function<add_fun_var> add_fun_var_f;

  template <typename Iterator>
  functions_grammar<Iterator>::functions_grammar(variable_map& var_map,
                                                 std::stringstream& error_msgs)
      : functions_grammar::base_type(functions_r),
        var_map_(var_map),
        functions_declared_(),
        functions_defined_(),
        error_msgs_(error_msgs),
        statement_g(var_map_,error_msgs_),
        bare_type_g(var_map_,error_msgs_)
  {
      using boost::spirit::qi::_1;
      using boost::spirit::qi::char_;
      using boost::spirit::qi::eps;
      using boost::spirit::qi::lexeme;
      using boost::spirit::qi::lit;
      using boost::spirit::qi::no_skip;
      using boost::spirit::qi::_pass;
      using boost::spirit::qi::_val;

      using boost::spirit::qi::labels::_a;
      using boost::spirit::qi::labels::_b;
      using boost::spirit::qi::labels::_r1;
      using boost::spirit::qi::labels::_r2;

      using boost::spirit::qi::on_error;
      using boost::spirit::qi::fail;
      using boost::spirit::qi::rethrow;
      using namespace boost::spirit::qi::labels;

      functions_r.name("function declarations and definitions");
      functions_r 
        %= lit("functions") 
        >> lit('{')
        >> *function_r
        >> lit('}')[ validate_declarations_f(_pass, 
                                             boost::phoenix::ref(functions_declared_),
                                             boost::phoenix::ref(functions_defined_),
                                             boost::phoenix::ref(error_msgs_) ) ]
        ;
      // locals: _a = allow sampling, _b = origin (function, rng/lp)
      function_r.name("function declaration or definition");
      function_r
        %= bare_type_g[ set_void_function_f(_1,_b, _pass, 
                                            boost::phoenix::ref(error_msgs_)) ]
        > identifier_r[ set_allows_sampling_origin_f(_1,_a,_b) ]
        > lit('(')
        > arg_decls_r
        > lit(')')
        > eps [ scope_lp_f(boost::phoenix::ref(var_map_)) ]
        > statement_g(_a,_b,true)
        > eps [ unscope_variables_f(_val,
                                     boost::phoenix::ref(var_map_)) ]
        > eps [ validate_return_type_f(_val,_pass,
                                        boost::phoenix::ref(error_msgs_)) ]
        > eps [ add_function_signature_f(_val,_pass,
                                          boost::phoenix::ref(functions_declared_),
                                          boost::phoenix::ref(functions_defined_),
                                          boost::phoenix::ref(error_msgs_) ) ]
        ;
      
      arg_decls_r.name("function argument declaration sequence");
      arg_decls_r
        %= arg_decl_r % ','
        | eps
        ;

      arg_decl_r.name("function argument declaration");
      arg_decl_r 
        %= bare_type_g [ validate_non_void_arg_f(_1, _pass, 
                                                 boost::phoenix::ref(error_msgs_)) ]
        > identifier_r
        > eps[ add_fun_var_f(_val,_pass,
                              boost::phoenix::ref(var_map_),
                              boost::phoenix::ref(error_msgs_)) ]
        ;

      identifier_r.name("identifier");
      identifier_r
        %= lexeme[char_("a-zA-Z") 
                   >> *char_("a-zA-Z0-9_.")];

    }

  }
}
#endif

