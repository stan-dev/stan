#ifndef STAN__GM__PARSER__FUNCTIONS_GRAMMAR_HPP__
#define STAN__GM__PARSER__FUNCTIONS_GRAMMAR_HPP__

#include <boost/spirit/include/qi.hpp>
#include <stan/gm/ast.hpp>
#include <stan/gm/grammars/bare_type_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "boost/spirit/home/qi/nonterminal/grammar.hpp"
#include "boost/spirit/home/qi/nonterminal/rule.hpp"
#include "boost/spirit/home/support/nonterminal/locals.hpp"

namespace stan { 

  namespace gm {

template <typename Iterator> struct whitespace_grammar;

    template <typename Iterator>
    struct functions_grammar 
      : boost::spirit::qi::grammar<Iterator,
                                   std::vector<function_decl_def>(),
                                   whitespace_grammar<Iterator> > {

      // global variable info
      variable_map& var_map_;

      // local info to keep track of which functions declared defined
      // so far
      std::set<std::pair<std::string, 
                         function_signature_t> > functions_declared_;
      std::set<std::pair<std::string, 
                         function_signature_t> > functions_defined_;

      std::stringstream& error_msgs_;
      
      // grammars
      statement_grammar<Iterator> statement_g;
      bare_type_grammar<Iterator> bare_type_g;
      
      // constructor
      functions_grammar(variable_map& var_map,
                        std::stringstream& error_msgs);

      // rules
      boost::spirit::qi::rule<Iterator, 
                              std::vector<function_decl_def>(),
                              whitespace_grammar<Iterator> > 
      functions_r;

      boost::spirit::qi::rule<Iterator, 
                              boost::spirit::qi::locals<bool,int>, 
                              function_decl_def(),
                              whitespace_grammar<Iterator> > 
      function_r;

      boost::spirit::qi::rule<Iterator, 
                              std::vector<arg_decl>(),
                              whitespace_grammar<Iterator> > 
      arg_decls_r;

      boost::spirit::qi::rule<Iterator, 
                              arg_decl(),
                              whitespace_grammar<Iterator> > 
      arg_decl_r;

      boost::spirit::qi::rule<Iterator, 
                              std::string(), 
                              whitespace_grammar<Iterator> > 
      identifier_r;

    };
                               

  }
}


#endif
