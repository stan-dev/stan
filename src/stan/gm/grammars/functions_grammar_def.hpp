#ifndef STAN__GM__PARSER__FUNCTIONS__GRAMMAR_DEF__HPP__
#define STAN__GM__PARSER__FUNCTIONS__GRAMMAR_DEF__HPP__

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


// BOOST_FUSION_ADAPT_STRUCT(stan::gm::function_decl_defs,
//                          (stan::gm::function_decl_def, decl_defs_) );

BOOST_FUSION_ADAPT_STRUCT(stan::gm::function_decl_def,
                          (std::string, name_) );

namespace stan {

  namespace gm {

  template <typename Iterator>
  functions_grammar<Iterator>::functions_grammar(variable_map& var_map,
                                                 std::stringstream& error_msgs)
      : functions_grammar::base_type(functions_r),
        var_map_(var_map),
        error_msgs_(error_msgs),
        statement_g(var_map_,error_msgs_)
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
        >> lit('}')
        ;
      
      function_r.name("function declaration or definition");
      function_r
        %= identifier_r
        >> lit(';')
        ;

      identifier_r.name("identifier");
      identifier_r
        %= (lexeme[char_("a-zA-Z") 
                   >> *char_("a-zA-Z0-9_.")]);

      
    }

  }
}
#endif

