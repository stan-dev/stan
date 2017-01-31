#ifndef STAN_LANG_COMPILER_HPP
#define STAN_LANG_COMPILER_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator.hpp>
#include <stan/lang/parser.hpp>

#include <iostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Read a Stan model specification from the
     * specified input, parse it, and write the C++ code for it
     * to the specified output.
     *
     * @param msgs Output stream for warning messages
     * @param stan_lang_in Stan model specification
     * @param cpp_out C++ code output stream
     * @param model_name Name of model class
     * @param allow_undefined Permit undefined functions?
     *
     * @return <code>false</code> if code could not be generated
     *    due to syntax error in the Stan model;
     *    <code>true</code> otherwise.
     */
    bool compile(std::ostream* msgs,
                 std::istream& stan_lang_in,
                 std::ostream& cpp_out,
                 const std::string& model_name,
                 const bool allow_undefined = false) {
      program prog;
      bool parsed_ok = parse(msgs, stan_lang_in,
                             model_name, prog, allow_undefined);
      if (!parsed_ok)
        return false;  // syntax error in program
      generate_cpp(prog, model_name, cpp_out);
      return true;
    }


  }
}
#endif
