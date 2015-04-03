#ifndef STAN_LANG_GRAMMARS_BARE_TYPE_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_BARE_TYPE_GRAMMAR_HPP

#include <string>
#include <sstream>
#include <vector>

#include <boost/spirit/include/qi.hpp>

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>
#include <stan/lang/grammars/statement_grammar.hpp>

namespace stan { 

  namespace lang {

    template <typename Iterator>
    struct bare_type_grammar
      : boost::spirit::qi::grammar<Iterator,
                                   expr_type(),
                                   whitespace_grammar<Iterator> > {

      // global info for function defs
      variable_map& var_map_;
      std::stringstream& error_msgs_;
      
      bare_type_grammar(variable_map& var_map,
                        std::stringstream& error_msgs);

      boost::spirit::qi::rule<Iterator, 
                              expr_type(),
                              whitespace_grammar<Iterator> > 
      bare_type_r;

      boost::spirit::qi::rule<Iterator, 
                              base_expr_type(),
                              whitespace_grammar<Iterator> > 
      type_identifier_r;

      boost::spirit::qi::rule<Iterator, 
                              size_t(),
                              whitespace_grammar<Iterator> > 
      array_dims_r;

      boost::spirit::qi::rule<Iterator,
                              boost::spirit::qi::unused_type,
                              whitespace_grammar<Iterator> >
      end_bare_types_r;

      
    };
  }
}

#endif
