#ifndef STAN__GM__PARSER__FUNCTIONS__GRAMMAR__HPP__
#define STAN__GM__PARSER__FUNCTIONS__GRAMMAR__HPP__

#include <string>
#include <sstream>
#include <vector>

#include <boost/spirit/include/qi.hpp>

#include <stan/gm/ast.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>

namespace stan { 

  namespace gm {

    template <typename Iterator>
    struct functions_grammar 
      : boost::spirit::qi::grammar<Iterator,
                                   std::vector<function_decl_def>(),
                                   whitespace_grammar<Iterator> > {

      // global info for function defs
      variable_map& var_map_;
      std::stringstream& error_msgs_;
      
      // grammars
      statement_grammar<Iterator> statement_g;
      
      // constructor
      functions_grammar(variable_map& var_map,
                        std::stringstream& error_msgs);

      // rules
      boost::spirit::qi::rule<Iterator, 
                              std::vector<function_decl_def>(),
                              whitespace_grammar<Iterator> > 
      functions_r;

      boost::spirit::qi::rule<Iterator, 
                              function_decl_def(),
                              whitespace_grammar<Iterator> > 
      function_r;

      boost::spirit::qi::rule<Iterator, 
                              std::string(), 
                              whitespace_grammar<Iterator> > 
      identifier_r;
      
    };
                               

  }
}


#endif
