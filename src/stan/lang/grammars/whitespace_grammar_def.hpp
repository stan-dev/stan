#ifndef STAN__LANG__PARSER__WHITESPACE_GRAMMAR_DEF__HPP
#define STAN__LANG__PARSER__WHITESPACE_GRAMMAR_DEF__HPP

#include <boost/spirit/include/qi.hpp>

#include <stan/lang/grammars/whitespace_grammar.hpp>

namespace stan {

  namespace lang {

    template <typename Iterator>
    whitespace_grammar<Iterator>::whitespace_grammar()
      : whitespace_grammar::base_type(whitespace)
    {
      using boost::spirit::qi::omit;
      using boost::spirit::qi::char_;
      using boost::spirit::qi::eol;
      whitespace
        = ( ( omit["/*"]
              >> *(char_ - "*/") )
            > omit["*/"] )
        | ( omit["//"] >> *(char_ - eol) )
        | ( omit["#"] >> *(char_ - eol) )
        | boost::spirit::ascii::space_type()
        ;
    }

  }

}



#endif
