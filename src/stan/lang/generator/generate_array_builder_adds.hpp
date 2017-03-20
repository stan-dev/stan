#ifndef STAN_LANG_GENERATOR_GENERATE_ARRAY_BUILDER_ADDS_HPP
#define STAN_LANG_GENERATOR_GENERATE_ARRAY_BUILDER_ADDS_HPP

#include <stan/lang/ast.hpp>
#include <stan/lang/generator/generate_expression.hpp>
#include <ostream>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Recursive helper function for array, matrix, and row_vector expressions
     * which generates chain of calls to math lib array_builder add function
     * for each of the contained elements.
     *
     * @param[in] elements vector of expression elements to generate
     * @param[in] user_facing true if generation is to read by user, false
     * for code generation in C++
     * @param[in] is_var_context true if generation in parameter var
     * context, false for data context
     * @param[in,out] o stream for generating
     */
    void generate_array_builder_adds(const std::vector<expression>& elements,
                                     bool user_facing,
                                     bool is_var_context,
                                     std::ostream& o) {
      for (size_t i = 0; i < elements.size(); ++i) {
        o << ".add(";
        generate_expression(elements[i], user_facing, is_var_context, o);
        o << ")";
      }
    }

  }
}
#endif
