// using boost::spirit::qi::fail;
// using boost::spirit::qi::locals;

#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
#include <boost/spirit/include/phoenix_bind.hpp>
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/support_multi_pass.hpp>

using std::set;
using std::string;
using std::vector;
using boost::phoenix::bind;
using boost::phoenix::cref;
using boost::phoenix::val;
using boost::spirit::qi::char_;
using boost::spirit::qi::eps;
using boost::spirit::qi::grammar;
using boost::spirit::qi::lexeme;
using boost::spirit::qi::on_error;
using boost::spirit::qi::rethrow;
using boost::spirit::qi::rule;
using boost::spirit::qi::_pass;
using boost::spirit::qi::_1;
using boost::spirit::qi::ascii::space_type;

template <typename Iterator> 
struct test_gram : grammar<Iterator, vector<string>(),
                           space_type> {
  test_gram() : test_gram::base_type(identifier_seq), ok_id_(true) {
    identifier_seq 
      %= +(identifier[bind(&test_gram::add_symbol,this,_1)] 
	   > eps[_pass = cref(ok_id_)]); 
    
    identifier 
      %= lexeme[+char_("a-zA-Z")];

    on_error<rethrow>(identifier_seq,
		      std::cout 
		      << val("duplicate symbol=\"")
		      << cref(symbol_) // bind(&test_gram::symbol,this) 
		      << "\"" 
		      << std::endl);
  }
  void add_symbol(const std::string& symbol) {
    if (symbol_table_.insert(symbol).second)
      return; 
    ok_id_ = false;
    symbol_ = symbol;
  }
  string symbol_;
  bool ok_id_;
  set<string> symbol_table_;
  rule<Iterator, vector<string>(),space_type> identifier_seq;
  rule<Iterator, string(), space_type> identifier;
};


// adapted from Spirit Qi blog
bool parse(std::istream& input, 
	   const std::string& filename, 
	   std::vector<std::string>& result) {

  namespace classic = boost::spirit::classic;

  // iterate over stream input
  typedef std::istreambuf_iterator<char> base_iterator_type;
  base_iterator_type in_begin(input);
      
  // convert input iterator to forward iterator, usable by spirit parser
  typedef boost::spirit::multi_pass<base_iterator_type> forward_iterator_type;
  forward_iterator_type fwd_begin = boost::spirit::make_default_multi_pass(in_begin);
  forward_iterator_type fwd_end;
      
  // wrap forward iterator with position iterator, to record the position
  typedef classic::position_iterator2<forward_iterator_type> pos_iterator_type;
  pos_iterator_type position_begin(fwd_begin, fwd_end, filename);
  pos_iterator_type position_end;
      
  test_gram<pos_iterator_type> test_grammar;
  boost::spirit::qi::ascii::space_type whitespace_grammar;
      
  bool success = 0;
  try {
    success = boost::spirit::qi::phrase_parse(position_begin, 
					      position_end,
					      test_grammar,
					      whitespace_grammar,
					      result);
  } catch (const boost::spirit::qi::expectation_failure<pos_iterator_type>& e) {
    const boost::spirit::classic::file_position_base<std::string>& pos = e.first.get_position();
    std::stringstream msg;
    msg << "parse error at file " 
	<< pos.file 
	<< " line " 
	<< pos.line 
	<< " column " 
	<< pos.column 
	<< std::endl 
	<< e.first.get_currentline() 
	<< std::endl;
    for (int i = 2; i < pos.column; ++i)
      msg << ' ';
    // << std::setw(pos.column) 
    msg << "^-- here";
    throw std::runtime_error(msg.str());
  }
  return success && (position_begin == position_end); // want to consume ALL input
}


int main() {
  std::cout << "START." << std::endl;
  static int SUCCESS_RC = 0;
  static int EXCEPTION_RC = -1;
  static int PARSE_FAIL_RC = -2;

  std::vector<std::string> symbols;
  try {
    bool succeeded = parse(std::cin, "STDIN", symbols); 
    if (!succeeded) {
      std::cout << "PARSE FAIL." << std::endl;
      return PARSE_FAIL_RC;
    }
  } catch(const std::exception& e) {
    std::cerr << "  EXCEPTION. " << e.what() << std::endl;
    return EXCEPTION_RC;
  }

  std::cout << "SUCCESS." << std::endl;

  std::cout << "return val" << std::endl;
  for (unsigned int i = 0; i < symbols.size(); ++i)
    std::cout << i << ". " << symbols[i] << std::endl;
  
  return SUCCESS_RC;
}

