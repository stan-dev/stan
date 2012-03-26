#ifndef __STAN__GM__PARSER__WHITESPACE_GRAMMAR_HPP__
#define __STAN__GM__PARSER__WHITESPACE_GRAMMAR_HPP__

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
