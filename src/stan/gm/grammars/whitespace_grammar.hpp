#ifndef STAN__GM__PARSER__WHITESPACE_GRAMMAR_HPP
#define STAN__GM__PARSER__WHITESPACE_GRAMMAR_HPP

#include <boost/spirit/include/qi.hpp>

namespace stan { 

  namespace gm {

    template <typename Iterator>
    struct whitespace_grammar 
      : public boost::spirit::qi::grammar<Iterator> {
    public:
      whitespace_grammar();
      // private:
      boost::spirit::qi::rule<Iterator> whitespace;
    };


  }

}



#endif
