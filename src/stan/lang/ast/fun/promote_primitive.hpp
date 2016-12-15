#ifndef STAN_LANG_AST_FUN_PROMOTE_PRIMITIVE_HPP
#define STAN_LANG_AST_FUN_PROMOTE_PRIMITIVE_HPP

#include <ostream>

namespace stan {
  namespace lang {

    struct expr_type;

    /**
     *
     * @param et expression type
     * @return promoted expression type
     */
    expr_type promote_primitive(const expr_type& et);

    /**
     *
     * @param et1 first expression type
     * @param et2 second expression type
     * @return promoted expression type
     */
    expr_type promote_primitive(const expr_type& et1, const expr_type& et2);

  }
}
#endif
