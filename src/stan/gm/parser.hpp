#ifndef STAN__GM__PARSER__PARSER__HPP
#define STAN__GM__PARSER__PARSER__HPP

#include <boost/lexical_cast.hpp>
#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_numeric.hpp>
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

#include <boost/spirit/include/version.hpp>
#include <boost/spirit/home/support/iterators/line_pos_iterator.hpp>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <istream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>

#include <stan/gm/ast.hpp>

#include <stan/gm/grammars/program_grammar.hpp>
#include <stan/gm/grammars/whitespace_grammar.hpp>
#include <stan/gm/grammars/expression_grammar.hpp>
#include <stan/gm/grammars/expression07_grammar.hpp>
#include <stan/gm/grammars/statement_grammar.hpp>
#include <stan/gm/grammars/var_decls_grammar.hpp>

namespace stan {

  namespace gm {

    bool is_space(char c) {
      return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    }

    bool is_nonempty(std::string& s) {
      for (size_t i = 0; i < s.size(); ++i)
        if (!is_space(s[i]))
          return true;
      return false;
    }

    inline bool parse(std::ostream* output_stream,
                      std::istream& input, 
                      const std::string& filename, 
                      const std::string& model_name,
                      program& result) {

      using boost::spirit::multi_pass;
      using boost::spirit::make_default_multi_pass;
      using std::istreambuf_iterator;

      using boost::spirit::qi::expectation_failure;
      using boost::spirit::qi::phrase_parse;

      using boost::phoenix::construct;
      using boost::phoenix::val;

      stan::gm::function_signatures::reset_sigs();

      std::ostringstream buf;
      buf << input.rdbuf();
      std::string stan_string = buf.str();

      typedef std::string::const_iterator input_iterator;
      typedef boost::spirit::line_pos_iterator<input_iterator> lp_iterator;

      lp_iterator fwd_begin = lp_iterator (stan_string.begin());
      lp_iterator fwd_end = lp_iterator (stan_string.end());

      program_grammar<lp_iterator> prog_grammar(model_name);
      whitespace_grammar<lp_iterator> whitesp_grammar;
      
      bool parse_succeeded = false;
      try {
        parse_succeeded = phrase_parse(fwd_begin, 
                                       fwd_end,
                                       prog_grammar,
                                       whitesp_grammar,
                                       result);
        std::string diagnostics = prog_grammar.error_msgs_.str();
        if (output_stream && is_nonempty(diagnostics)) {
          *output_stream << "DIAGNOSTIC(S) FROM PARSER:" 
                         << std::endl
                         << diagnostics 
                         << std::endl;
        }
      } catch (const expectation_failure<lp_iterator>& e) {

        std::stringstream msg;
        std::string diagnostics = prog_grammar.error_msgs_.str();
        if (output_stream && is_nonempty(diagnostics)) {
          msg << "SYNTAX ERROR, MESSAGE(S) FROM PARSER:"
              << std::endl
              << diagnostics
              << std::endl;
        }

        throw std::invalid_argument(msg.str());

      } catch (const std::runtime_error& e) {
        std::stringstream msg;
        msg << "PROGRAM ERROR, MESSAGE(S) FROM PARSER:"
            << std::endl
            << prog_grammar.error_msgs_.str()
            << std::endl;

        throw std::invalid_argument(msg.str());
      }
      
      bool consumed_all_input = (fwd_begin == fwd_end); 
      bool success = parse_succeeded && consumed_all_input;

      if (!success) {      

        std::stringstream msg;
        if (!parse_succeeded)
          msg << "PARSE FAILED." << std::endl; 
        if (!consumed_all_input) {
          // get rest of program
          std::basic_stringstream<char> unparsed_non_ws;
          unparsed_non_ws << boost::make_iterator_range(fwd_begin, fwd_end);
          // get next two lines (if possible)
          msg << "PARSING HALTED AT LINE "
              << get_line(fwd_begin)
              << std::endl
              << "UNPARSED STAN PROGRAM: "
              << std::endl
              << unparsed_non_ws.str()
              << std::endl;
        }
        msg << std::endl << prog_grammar.error_msgs_.str() << std::endl;
        throw std::invalid_argument(msg.str());
      }
      return true;
    }


  }
}

#endif
