#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
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

#include <iomanip>
#include <iostream>
#include <istream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>

namespace foo {

  using namespace boost::spirit;

  template <typename Iterator>
  class whitespace_grammar : public qi::grammar<Iterator> {
  public:
    whitespace_grammar() : whitespace_grammar::base_type(whitespace) {
      whitespace 
        = ( qi::omit["/*"] >> *(qi::char_ - "*/") > qi::omit["*/"] )
        | ( qi::omit["//"] >> *(qi::char_ - qi::eol) )
        | ( qi::omit["#"] >> *(qi::char_ - qi::eol) )
        | ascii::space_type()
        ;
    }
  private:
    qi::rule<Iterator> whitespace;
  };

  template <typename Iterator>
  class big_grammar : public qi::grammar<Iterator,
                                          std::string(),
                                          whitespace_grammar<Iterator> > {
  public:
    big_grammar()
      : big_grammar::base_type(root_r) {

      using qi::_val;
      using qi::_1;
      using qi::_pass;
      using qi::double_;
      using qi::int_;
      using boost::spirit::qi::eps;
      using namespace qi::labels;
      
      root_r %= qi::lit("foo");
    }
  private:
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > root_r;
  };

  bool parse(std::istream& input, 
             const std::string& filename, 
             std::string& result) {
   
    namespace classic = boost::spirit::classic;

    // iterate over stream input
    typedef std::istreambuf_iterator<char> base_iterator_type;

    base_iterator_type in_begin(input);
      
    // convert input iterator to forward iterator, usable by spirit parser
    typedef boost::spirit::multi_pass<base_iterator_type> 
      forward_iterator_type;

    forward_iterator_type fwd_begin 
      = boost::spirit::make_default_multi_pass(in_begin);
    forward_iterator_type fwd_end;
      
    // wrap forward iterator with position iterator, to record the position
    typedef classic::position_iterator2<forward_iterator_type> 
      pos_iterator_type;

    pos_iterator_type position_begin(fwd_begin, fwd_end, filename);
    pos_iterator_type position_end;
      
    foo::big_grammar<pos_iterator_type> prog_grammar;
    foo::whitespace_grammar<pos_iterator_type> whitesp_grammar;
      
    bool success = qi::phrase_parse(position_begin, 
                                    position_end,
                                    prog_grammar,
                                    whitesp_grammar,
                                    result);
    return success;
  }

}

int main(int argc, char* argv[]) {
  std::string text = argv[1];
  std::stringstream text_stream(text);
  std::string result;
  bool success = foo::parse(text_stream,"DUMMY",result);
  std::cout << "success=" << success << std::endl;
  std::cout << "result=" << result << std::endl;
}
                  
