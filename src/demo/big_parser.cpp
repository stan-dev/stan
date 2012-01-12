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
  class sub_grammar : public qi::grammar<Iterator,
                                         std::string(),
                                         whitespace_grammar<Iterator> > {
  public:
    sub_grammar()
      : sub_grammar::base_type(root_r) {
      root_r %= qi::lit("foo");
    }
  private:
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > root_r;
  };

  template <typename Iterator>
  class big_grammar : public qi::grammar<Iterator,
                                          std::string(),
                                          whitespace_grammar<Iterator> > {
  public:
    big_grammar()
      : big_grammar::base_type(root_r) {

      root_r %= *sub_r;
      rule1_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule2_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule3_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule4_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule5_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule6_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule7_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule8_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule9_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule10_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule11_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule12_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule13_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule14_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule15_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule16_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule17_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule18_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule29_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule20_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule21_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule22_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule23_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule24_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule25_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule26_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule27_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule28_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule29_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule30_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule31_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule32_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule33_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule34_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule35_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule36_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule37_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule38_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule39_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule40_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule41_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule42_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule43_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule44_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule45_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule46_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule47_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule48_r %= *qi::lit("bar") >> -qi::lit("baz");
      rule49_r %= *qi::lit("bar") >> -qi::lit("baz");
    }
  private:
    sub_grammar<Iterator> sub_r;
    
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > root_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule1_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule2_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule3_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule4_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule5_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule6_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule7_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule8_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule9_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule10_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule11_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule12_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule13_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule14_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule15_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule16_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule17_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule18_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule19_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule20_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule21_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule22_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule23_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule24_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule25_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule26_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule27_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule28_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule29_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule30_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule31_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule32_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule33_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule34_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule35_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule36_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule37_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule38_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule39_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule40_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule41_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule42_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule43_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule44_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule45_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule46_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule47_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule48_r;
    qi::rule<Iterator, std::string(), whitespace_grammar<Iterator> > rule49_r;
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
                  
