#ifndef STAN_LANG_GENERATOR_GENERATE_WRITE_ARRAY_METHOD_HPP
#define STAN_LANG_GENERATOR_GENERATE_WRITE_ARRAY_METHOD_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_catch_throw_located.hpp>
#include <stan/lang/generator/generate_comment.hpp>
#include <stan/lang/generator/generate_local_var_decls.hpp>
#include <stan/lang/generator/generate_statements.hpp>
#include <stan/lang/generator/generate_try.hpp>
#include <stan/lang/generator/generate_validate_var_decls.hpp>
#include <stan/lang/generator/generate_void_statement.hpp>
#include <stan/lang/generator/write_array_visgen.hpp>
#include <stan/lang/generator/write_array_vars_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>
#include <string>

namespace stan {
  namespace lang {

    /**
     * Generate the <code>write_array</code> method for the specified
     * program, with specified model name to the specified stream.
     *
     * @param[in] prog program from which to generate
     * @param[in] model_name name of model
     * @param[in,out] o stream for generating
     */
    void generate_write_array_method(const program& prog,
                                     const std::string& model_name,
                                     std::ostream& o) {
      o << INDENT << "template <typename RNG>" << EOL;
      o << INDENT << "void write_array(RNG& base_rng__," << EOL;
      o << INDENT << "                 std::vector<double>& params_r__," << EOL;
      o << INDENT << "                 std::vector<int>& params_i__," << EOL;
      o << INDENT << "                 std::vector<double>& vars__," << EOL;
      o << INDENT << "                 bool include_tparams__ = true," << EOL;
      o << INDENT << "                 bool include_gqs__ = true," << EOL;
      o << INDENT
        << "                 std::ostream* pstream__ = 0) const {" << EOL;
      o << INDENT2 << "typedef double local_scalar_t__;" << EOL2;

      o << INDENT2 << "vars__.resize(0);" << EOL;
      o << INDENT2
        << "stan::io::reader<local_scalar_t__> in__(params_r__,params_i__);"
        << EOL;
      o << INDENT2 << "static const char* function__ = \""
        << model_name << "_namespace::write_array\";" << EOL;
      generate_void_statement("function__", 2, o);

      // declares, reads, and sets parameters
      generate_comment("read-transform, write parameters", 2, o);
      write_array_visgen vis(o);
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);


      // writes parameters
      write_array_vars_visgen vis_writer(2, o);
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        boost::apply_visitor(vis_writer, prog.parameter_decl_[i].decl_);
      o << EOL;

      o << INDENT2 << "if (!include_tparams__) return;"
        << EOL;
      generate_comment("declare and define transformed parameters", 2, o);
      o << INDENT2 <<  "double lp__ = 0.0;" << EOL;
      generate_void_statement("lp__", 2, o);
      o << INDENT2 << "stan::math::accumulator<double> lp_accum__;" << EOL2;

      o << INDENT2
        << "local_scalar_t__ DUMMY_VAR__"
        << "(std::numeric_limits<double>::quiet_NaN());"
        << EOL;
      o << INDENT2 << "(void) DUMMY_VAR__;  // suppress unused var warning"
        << EOL2;

      generate_try(2, o);
      generate_local_var_decls(prog.derived_decl_.first, 3, o);
      o << EOL;
      bool include_sampling = false;
      generate_statements(prog.derived_decl_.second, 3, o,
                          include_sampling);
      o << EOL;

      generate_comment("validate transformed parameters", 3, o);
      generate_validate_var_decls(prog.derived_decl_.first, 3, o);
      o << EOL;

      generate_comment("write transformed parameters", 3, o);
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i)
        boost::apply_visitor(vis_writer, prog.derived_decl_.first[i].decl_);
      o << EOL;

      o << INDENT3 << "if (!include_gqs__) return;"
        << EOL;
      generate_comment("declare and define generated quantities", 3, o);
      generate_local_var_decls(prog.generated_decl_.first, 3, o);

      o << EOL;
      generate_statements(prog.generated_decl_.second,
                          3, o, include_sampling);
      o << EOL;

      generate_comment("validate generated quantities", 3, o);
      generate_validate_var_decls(prog.generated_decl_.first, 3, o);
      o << EOL;

      generate_comment("write generated quantities", 3, o);
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i)
        boost::apply_visitor(vis_writer, prog.generated_decl_.first[i].decl_);
      if (prog.generated_decl_.first.size() > 0)
        o << EOL;
      generate_catch_throw_located(2, o);

      o << INDENT << "}" << EOL2;

      o << INDENT << "template <typename RNG>" << EOL;
      o << INDENT << "void write_array(RNG& base_rng," << EOL;
      o << INDENT
        << "                 Eigen::Matrix<double,Eigen::Dynamic,1>& params_r,"
        << EOL;
      o << INDENT
        << "                 Eigen::Matrix<double,Eigen::Dynamic,1>& vars,"
        << EOL;
      o << INDENT << "                 bool include_tparams = true," << EOL;
      o << INDENT << "                 bool include_gqs = true," << EOL;
      o << INDENT
        << "                 std::ostream* pstream = 0) const {" << EOL;
      o << INDENT
        << "  std::vector<double> params_r_vec(params_r.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < params_r.size(); ++i)" << EOL;
      o << INDENT << "    params_r_vec[i] = params_r(i);" << EOL;
      o << INDENT << "  std::vector<double> vars_vec;" << EOL;
      o << INDENT << "  std::vector<int> params_i_vec;" << EOL;
      o << INDENT
        << "  write_array(base_rng,params_r_vec,params_i_vec,"
        << "vars_vec,include_tparams,include_gqs,pstream);" << EOL;
      o << INDENT << "  vars.resize(vars_vec.size());" << EOL;
      o << INDENT << "  for (int i = 0; i < vars.size(); ++i)" << EOL;
      o << INDENT << "    vars(i) = vars_vec[i];" << EOL;
      o << INDENT << "}" << EOL2;
    }

  }
}
#endif
