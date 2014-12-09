#ifndef STAN__LANG__COMPILER_HPP
#define STAN__LANG__COMPILER_HPP

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
     * @param include_main Indicates whether to generate a main
     *    function
     * @param in_file_name Name of input file to use in error
     * messages; defaults to <code>input</code>.
     * @return <code>false</code> if code could not be generated
     *    due to syntax error in the Stan model; 
     *    <code>true</code> otherwise.
     * @param msgs
     * @param stan_lang_in
     * @param cpp_out
     * @param model_name
     * @param in_file_name
     */
    bool compile(std::ostream* msgs, // for warnings
                 std::istream& stan_lang_in,
                 std::ostream& cpp_out,
                 const std::string& model_name,
                 const std::string& in_file_name = "input") {
      program prog;
      bool parsed_ok = parse(msgs,stan_lang_in,in_file_name,model_name,prog);
      if (!parsed_ok) 
        return false; // syntax error in program
      generate_cpp(prog,model_name,cpp_out);
      return true;
    }
    

  }

}

#endif
