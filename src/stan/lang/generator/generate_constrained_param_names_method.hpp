#ifndef STAN_LANG_GENERATOR_GENERATE_CONSTRAINED_PARAM_NAMES_METHOD_HPP
#define STAN_LANG_GENERATOR_GENERATE_CONSTRAINED_PARAM_NAMES_METHOD_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/constrained_param_names_visgen.hpp>
#include <boost/variant/apply_visitor.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    /**
     * Generate the <code>constrained_param_names</code> method for
     * the specified program on the specified stream.
     *
     * @param[in] prog program from which to generate
     * @param[in,out] o stream for generating
     */
    void generate_constrained_param_names_method(const program& prog,
                                                 std::ostream& o) {
      o << EOL << INDENT
        << "void constrained_param_names("
        << "std::vector<std::string>& param_names__,"
        << EOL << INDENT
        << "                             bool include_tparams__ = true,"
        << EOL << INDENT
        << "                             bool include_gqs__ = true) const {"
        << EOL << INDENT2
        << "std::stringstream param_name_stream__;" << EOL;
      constrained_param_names_visgen vis(o);
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        boost::apply_visitor(vis, prog.parameter_decl_[i].decl_);
      o << EOL << INDENT2 << "if (!include_gqs__ && !include_tparams__) return;"
        << EOL;
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i)
        boost::apply_visitor(vis, prog.derived_decl_.first[i].decl_);
      o << EOL << INDENT2 << "if (!include_gqs__) return;" << EOL;
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i)
        boost::apply_visitor(vis, prog.generated_decl_.first[i].decl_);
      o << INDENT << "}" << EOL2;
    }

  }
}
#endif
