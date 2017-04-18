#ifndef STAN_LANG_GENERATOR_GENERATE_CONSTRUCTOR_HPP
#define STAN_LANG_GENERATOR_GENERATE_CONSTRUCTOR_HPP

#include <stan/io/program_reader.hpp>
#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_comment.hpp>
#include <stan/lang/generator/generate_located_statements.hpp>
#include <stan/lang/generator/generate_member_var_inits.hpp>
#include <stan/lang/generator/generate_set_param_ranges.hpp>
#include <stan/lang/generator/generate_validate_var_decls.hpp>
#include <stan/lang/generator/generate_var_resizing.hpp>
#include <stan/lang/generator/generate_void_statement.hpp>
#include <ostream>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate the constructors for the specified program with the
     * specified model name to the specified stream.
     *
     * @param[in] prog program from which to generate
     * @param[in] model_name name of model for class name
     * @param[in] history list of file includes and positions used to
     *   create text of program
     * @param[in,out] o stream for generating
     */
    void generate_constructor(const program& prog,
                              const std::string& model_name,
                              const std::vector<io::preproc_event>& history,
                              std::ostream& o) {
      // constructor without RNG or template parameter
      o << INDENT << model_name << "(stan::io::var_context& context__," << EOL;
      o << INDENT << "    std::ostream* pstream__ = 0)" << EOL;
      o << INDENT2 << ": prob_grad(0) {" << EOL;
      o << INDENT2 << "typedef boost::ecuyer1988 rng_t;" << EOL;
      o << INDENT2 << "rng_t base_rng(0);  // 0 seed default" << EOL;
      o << INDENT2 << "ctor_body(context__, base_rng, pstream__);" << EOL;
      o << INDENT << "}" << EOL2;
      // constructor with specified RNG
      o << INDENT << "template <class RNG>" << EOL;
      o << INDENT << model_name << "(stan::io::var_context& context__," << EOL;
      o << INDENT << "    RNG& base_rng__," << EOL;
      o << INDENT << "    std::ostream* pstream__ = 0)" << EOL;
      o << INDENT2 << ": prob_grad(0) {" << EOL;
      o << INDENT2 << "ctor_body(context__, base_rng__, pstream__);" << EOL;
      o << INDENT << "}" << EOL2;
      // body of constructor now in function
      o << INDENT << "template <class RNG>" << EOL;
      o << INDENT << "void ctor_body(stan::io::var_context& context__," << EOL;
      o << INDENT << "               RNG& base_rng__," << EOL;
      o << INDENT << "               std::ostream* pstream__) {" << EOL;

      // generate preproc events before first statement
      generate_comment("filling program I/O history for error reports", 2, o);

      for (size_t i = 0; i < history.size(); ++i)
        o << INDENT2 << "prog_reader__.add_event("
          << history[i].concat_line_num_
          << ", " << history[i]. line_num_
          << ", \"" << history[i].action_ << "\""
          << ", \"" << history[i].path_ << "\");" << std::endl;
      o << std::endl;

      o << INDENT2 << "current_statement_begin__ = -1;" << EOL2;
      o << INDENT2 << "static const char* function__ = \""
        << model_name << "_namespace::" << model_name << "\";" << EOL;
      generate_void_statement("function__", 2, o);
      o << INDENT2 << "size_t pos__;" << EOL;
      generate_void_statement("pos__", 2, o);
      o << INDENT2 << "std::vector<int> vals_i__;" << EOL;
      o << INDENT2 << "std::vector<double> vals_r__;" << EOL;
      o << INDENT2
        << "double DUMMY_VAR__(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      o << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;
      o << INDENT2 << "// initialize member variables" << EOL;
      generate_member_var_inits(prog.data_decl_, o);
      o << EOL;
      generate_comment("validate, data variables", 2, o);
      generate_validate_var_decls(prog.data_decl_, 2, o);
      generate_comment("initialize data variables", 2, o);
      generate_var_resizing(prog.derived_data_decl_.first, o);
      o << EOL;

      bool include_sampling = false;
      bool is_var_context = false;
      bool is_fun_return = false;

      // need to fix generate_located_statements
      generate_located_statements(prog.derived_data_decl_.second,
                                  2, o, include_sampling, is_var_context,
                                  is_fun_return);
      o << EOL;
      generate_comment("validate transformed data", 2, o);
      generate_validate_var_decls(prog.derived_data_decl_.first, 2, o);
      o << EOL;
      generate_comment("validate, set parameter ranges", 2, o);
      generate_set_param_ranges(prog.parameter_decl_, o);
      o << INDENT << "}" << EOL;
    }

  }
}
#endif
