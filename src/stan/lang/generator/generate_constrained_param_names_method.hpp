#ifndef STAN_LANG_GENERATOR_GENERATE_CONSTRAINED_PARAM_NAMES_METHOD_HPP
#define STAN_LANG_GENERATOR_GENERATE_CONSTRAINED_PARAM_NAMES_METHOD_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <ostream>
#include <vector>

// TODO:mitzi check indexing of array dims  vs. vector/matrix dims

namespace stan {
  namespace lang {

    /**
     * Return vector of size expressions for all dimensions
     * of a block_var_decl in the following order:
     * matrix cols (if matrix type),
     * matrix row / row_vector / vector length (if matrix/vec type),
     * array dim N through array dim 1
     *
     * @param[in] var_decl block_var_decl
     */
    std::vector<expression>
    get_block_var_dims(const block_var_decl decl) {
      std::vector<expression> dims;
      if (decl.type().bare_type().is_matrix_type()) {
        dims.push_back(decl.type().arg2());
        dims.push_back(decl.type().arg1());
      } else if (decl.type().bare_type().is_row_vector_type()
                 || decl.type().bare_type().is_vector_type()) {
        dims.push_back(decl.type().arg1());
      }
      std::vector<expression> ar_lens = decl.type().array_lens();
      for (size_t i = ar_lens.size(); i-- > 0; ) {
        dims.push_back(ar_lens[i]);
      }
      return dims;
    }

    void
    generate_param_names_array(size_t indent, std::ostream& o,
                               const block_var_decl var_decl);
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
      for (size_t i = 0; i < prog.parameter_decl_.size(); ++i)
        generate_param_names_array(1, o, prog.parameter_decl_[i]);
      o << EOL << INDENT2
        << "if (!include_gqs__ && !include_tparams__) return;" << EOL;
      o << EOL << INDENT2 << "if (include_tparams__) {"  << EOL;
      for (size_t i = 0; i < prog.derived_decl_.first.size(); ++i)
        generate_param_names_array(2, o, prog.derived_decl_.first[i]);
      o << INDENT2 << "}" << EOL2;
      o << EOL << INDENT2 << "if (!include_gqs__) return;" << EOL;
      for (size_t i = 0; i < prog.generated_decl_.first.size(); ++i)
        generate_param_names_array(1, o, prog.generated_decl_.first[i]);
      o << INDENT << "}" << EOL2;
    }

    /**
     * Generate the parameter names for the specified parameter variable.
     *
     * @param[in] indent level of indentation
     * @param[in,out] o stream for generating
     * @param[in] var_decl block_var_decl
     */
    void
    generate_param_names_array(size_t indent, std::ostream& o,
                               const block_var_decl var_decl) {
      std::vector<expression> dims = get_block_var_dims(var_decl);
      for (size_t i = dims.size(); i-- > 0; ) {
        generate_indent(indent + dims.size() - i, o);
        o << "for (int k_" << i << "__ = 1;"
           << " k_" << i << "__ <= ";
        generate_expression(dims[i].expr_, NOT_USER_FACING, o);
        o << "; ++k_" << i << "__) {" << EOL;  // begin (1)
      }
      generate_indent(indent + 1 + dims.size(), o);
      o << "param_name_stream__.str(std::string());" << EOL;

      generate_indent(indent + 1 + dims.size(), o);
      o << "param_name_stream__ << \"" << var_decl.name() << '"';

      for (size_t i = 0; i < dims.size(); ++i)
        o << " << '.' << k_" << i << "__";
      o << ';' << EOL;

      generate_indent(indent + 1 + dims.size(), o);
      o << "param_names__.push_back(param_name_stream__.str());" << EOL;

      // end for loop dims
      for (size_t i = 0; i < dims.size(); ++i) {
        generate_indent(indent + dims.size() - i, o);
        o << "}" << EOL;  // end (1)
      }
    }
  }
}
#endif
