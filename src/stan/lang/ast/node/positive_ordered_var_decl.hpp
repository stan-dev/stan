#ifndef STAN_LANG_AST_NODE_POSITIVE_ORDERED_VAR_DECL_HPP
#define STAN_LANG_AST_NODE_POSITIVE_ORDERED_VAR_DECL_HPP

#include <stan/lang/ast/node/base_var_decl.hpp>
#include <stan/lang/ast/node/expression.hpp>
#include <string>
#include <vector>

namespace stan {
  namespace lang {

    /**
     * Structure to hold the declaration of a positive ordered vector.
     */
    struct positive_ordered_var_decl : public base_var_decl {
      /**
       * Positive rdered vector size.
       */
      expression K_;

      /**
       * Construct a default positive ordered vector declaration.
       */
      positive_ordered_var_decl();

      /**
       * Construct a positive ordered vector declaration with the
       * specified size, name, and number of array dimensions.
       *
       * @param K size of positive ordered vector
       * @param name variable name
       * @param dims array dimension sizes
       * @param def definition
       */
      positive_ordered_var_decl(const expression& K,
                                const std::string& name,
                                const std::vector<expression>& dims,
                                const expression& def);
    };

  }
}
#endif
