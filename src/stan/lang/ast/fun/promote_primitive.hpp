#ifndef STAN_LANG_AST_FUN_PROMOTE_PRIMITIVE_HPP
#define STAN_LANG_AST_FUN_PROMOTE_PRIMITIVE_HPP

#include <ostream>

namespace stan {
  namespace lang {

    struct bare_expr_type;

    /**
     *
     * @param et expression type
     * @return promoted expression type
     */
    bare_expr_type promote_primitive(const bare_expr_type& et);

    /**
     *
     * @param et1 first expression type
     * @param et2 second expression type
     * @return promoted expression type
     */
    bare_expr_type promote_primitive(const bare_expr_type& et1,
                                     const bare_expr_type& et2);
  }
}
#endif
