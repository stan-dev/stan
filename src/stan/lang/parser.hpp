#ifndef STAN_LANG_PARSER_HPP
#define STAN_LANG_PARSER_HPP

#include <boost/spirit/home/support/iterators/line_pos_iterator.hpp>
#include <boost/spirit/include/qi.hpp>

#include <stan/lang/ast.hpp>
#include <stan/lang/grammars/program_grammar.hpp>
#include <stan/lang/grammars/whitespace_grammar.hpp>

#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <stdexcept>

namespace stan {

  namespace lang {

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
                      const std::string& model_name,
                      program& result,
                      const bool allow_undefined = false) {
      using boost::spirit::qi::expectation_failure;
      using boost::spirit::qi::phrase_parse;

      stan::lang::function_signatures::reset_sigs();

      std::ostringstream buf;
      buf << input.rdbuf();
      std::string stan_string = buf.str();

      typedef std::string::const_iterator input_iterator;
      typedef boost::spirit::line_pos_iterator<input_iterator> lp_iterator;

      lp_iterator fwd_begin = lp_iterator(stan_string.begin());
      lp_iterator fwd_end = lp_iterator(stan_string.end());

      program_grammar<lp_iterator> prog_grammar(model_name, allow_undefined);
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
              << std::endl
              << diagnostics;
        }
        if (output_stream) {
          std::stringstream ss;
          ss << e.what_;
          std::string e_what = ss.str();
          std::string angle_eps("<eps>");
          if (e_what != angle_eps)
            msg << "PARSER EXPECTED: "
                << e.what_
                << std::endl;
        }
        throw std::invalid_argument(msg.str());
      } catch (const std::exception& e) {
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
          std::basic_stringstream<char> unparsed_non_ws;
          unparsed_non_ws << boost::make_iterator_range(fwd_begin, fwd_end);
          msg << "PARSER EXPECTED: whitespace to end of file."
              << std::endl
              << "FOUND AT line "
              << get_line(fwd_begin)
              << ": "
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
