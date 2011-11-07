#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_function.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/spirit/home/phoenix/container.hpp>

#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace spirit = boost::spirit;
namespace ascii = spirit::ascii;
namespace classic = spirit::classic;
namespace phoenix = boost::phoenix;
namespace qi = spirit::qi;

using namespace qi::labels;

namespace symtab {

  struct add_id_struct {
    template <typename T1, typename T2, typename T3>
    struct result { typedef void type; };

    void operator()(const std::string& x, std::set<std::string>& xs, 
		    bool& pass) const {
      pass = xs.insert(x).second;
    }
  };
  phoenix::function<add_id_struct> add_id;
    
  using phoenix::insert;

  template <typename It>
  struct st_grammar : qi::grammar<It,
				  std::vector<std::string>(), 
				  ascii::space_type> {
    std::set<std::string> vars_;
    st_grammar() : st_grammar::base_type(ids_r) {

      name_r %= qi::lexeme[+qi::char_("a-zA-Z")];

      id_r %= name_r[add_id(_1,phoenix::ref(vars_),_a)]
	> qi::eps[_pass = _a];
      
      ids_r %= +(id_r > qi::lexeme[';']);

      qi::on_error<qi::rethrow>(id_r,
				std::cout << phoenix::val("DUP ID \"") 
				<< qi::_val << "\"" << std::endl);
      qi::on_error<qi::rethrow>(ids_r,
				std::cout << phoenix::val("EXPECTED ")
				<< _4 << std::endl);
    }
    qi::rule<It,std::string(),ascii::space_type> name_r;
    qi::rule<It,qi::locals<bool>,std::string(),ascii::space_type> id_r;
    qi::rule<It, std::vector<std::string>(),ascii::space_type> ids_r;
  };
}

void parse(const std::string& input) {
  std::cout << std::endl << "INPUT=" << '"' << input << '"' << std::endl;
  typedef std::istreambuf_iterator<char> base_iterator_type;
  typedef spirit::multi_pass<base_iterator_type> forward_iterator_type;
  typedef classic::position_iterator2<forward_iterator_type> pos_iterator_type;
  std::stringbuf sb(input);
  base_iterator_type in_begin(&sb);
  forward_iterator_type fwd_begin 
    = spirit::make_default_multi_pass(in_begin);
  forward_iterator_type fwd_end;
  pos_iterator_type position_begin(fwd_begin, fwd_end, "string");
  pos_iterator_type position_end;
  symtab::st_grammar<pos_iterator_type> st_grammar;
  ascii::space_type whitespace_grammar;
  std::vector<std::string> symbols;
  try {
    if (!qi::phrase_parse(position_begin, position_end,
			  st_grammar, whitespace_grammar, 
			  symbols) || position_begin != position_end) {
      std::cerr << "PARSE FAILED." << std::endl;
      return;
    }
  } catch (const qi::expectation_failure<pos_iterator_type>& e) {
    std::cerr << "ERROR: file " << e.first.get_position().file
	      << " line " << e.first.get_position().line 
	      << " column " << e.first.get_position().column
	      << std::endl << e.first.get_currentline() << std::endl;
    for (int i = 1; i < e.first.get_position().column; ++i) std::cerr << ' ';
    std::cerr << "^-- here" << std::endl;
    return;
  }
  for (unsigned int i = 0; i < symbols.size(); ++i)
    std::cout << symbols[i] << std::endl;
}

int main() {
  parse("a; b; c;");  // OK
  parse("a; b c;");   // BAD: missing semicolon after b
  parse("a; b; a;");  // BAD: duplicated symbol a
}
