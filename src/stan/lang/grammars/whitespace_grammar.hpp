#ifndef STAN_LANG_GRAMMARS_WHITESPACE_GRAMMAR_HPP
#define STAN_LANG_GRAMMARS_WHITESPACE_GRAMMAR_HPP

#include <boost/spirit/include/qi.hpp>

namespace stan {

  namespace lang {

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
