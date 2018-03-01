#ifndef STAN_LANG_GENERATOR_GENERATE_VALIDATE_TRANSFORMED_PARAMS_HPP
#define STAN_LANG_GENERATOR_GENERATE_VALIDATE_TRANSFORMED_PARAMS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/constants.hpp>
#include <stan/lang/generator/generate_comment.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Generate code to validate the specified transformed parameters,
     * generating at the specified indentation level to the specified
     * stream.
     *
     * @param[in] vs variable declarations
     * @param[in] indent indentation level
     * @param[in,out] o stream for generating
     */
    void generate_validate_transformed_params(const std::vector<block_var_decl>& vs,
                                              int indent, std::ostream& o) {
      for (size_t i = 0; i < vs.size(); ++i) {

        // setup - name, type, and var shape
        std::string var_name(vs[i].name());
        // unfold array type to get array element info
        block_var_type btype = (vs[i].type());
        if (btype.is_array_type())
          btype = btype.array_contains();
        // dimension sizes and type - array or matrix/vec rows, columns
        std::vector<expression> ar_lens(vs[i].type().array_lens());
        expression arg1 = btype.arg1();
        expression arg2 = btype.arg2();
        // use parallel arrays of index names, sizes for validate loop
        std::vector<std::string> idx_names;
        std::vector<expression> limits;
        for (size_t i = 0; i < ar_lens.size(); ++i) {
          std::stringstream ss_name;
          ss_name << "i" << i << "__";
          idx_names.push_back(ss_name.str());
          limits.push_back(ar_lens[i]);
        }
        if (!is_nil(arg1)) {
          std::stringstream ss_name;
          ss_name << "i" << idx_names.size() << "__";
          idx_names.push_back(ss_name.str());
          limits.push_back(arg1);
        }          
        if (!is_nil(arg2)) {
          std::stringstream ss_name;
          ss_name << "i" << idx_names.size() << "__";
          idx_names.push_back(ss_name.str());
          limits.push_back(arg2);
        }          
        
        // validate loop
        // indexes innermost scalar element of (array of) matrix/vec types
        for (size_t i = 0; i < idx_names.size(); ++i) {
          generate_indent(indent + i, o);
          o << "for (int " << idx_names[i] << " = 0; "
            << idx_names[i] << " < ";
          generate_expression(limits[i], NOT_USER_FACING, o);
          o << "; ++" << idx_names[i] << ")" << EOL;
        }
        // validate stmt
        generate_indent(indent + idx_names.size(), o);
        o << "if (stan::math::is_uninitialized(" << var_name;
        for (size_t i = 0; i < ar_lens.size(); ++i)
          o << "[" << idx_names[i] << "]";
        if (!is_nil(arg1)) {
          o << "(" << idx_names[ar_lens.size()];
          if (!is_nil(arg2))
            o << ", " << idx_names[ar_lens.size()+1];
          o << ")";
        }
        o << ")) {\n" << EOL;
        generate_indent(indent + idx_names.size() + 1, o);
        o << "std::stringstream msg__;" << EOL;
        generate_indent(indent + idx_names.size() + 1, o);
        o << "msg__ << \"Undefined transformed parameter: "
          << var_name;
        if (idx_names.size() > 0)
          o << " << '[' << ";
        for (size_t i = 0; i < idx_names.size(); ++i)
          o << " << '[' << " << idx_names[i] << " << '['";
        o << ";" << EOL;
        generate_indent(indent + idx_names.size() + 1, o);
        o << "throw std::runtime_error(msg__.str());";
        generate_indent(indent + idx_names.size(), o);
        o << "}" << EOL;
        // close loop
        for (size_t i = idx_names.size(); i > 0; --i) {
          generate_indent(indent + i - 1, o);
          o << "}" << EOL;
        }
        o << EOL;
      }
    }

  }
}
#endif
