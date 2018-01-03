#ifndef STAN_LANG_AST_FUN_PROMOTE_PRIMITIVE_HPP
#define STAN_LANG_AST_FUN_PROMOTE_PRIMITIVE_HPP

#include <ostream>

namespace stan {
  namespace lang {

    struct bare_expr_type;

    /**
     * Int and double types promote (sic) to same;
     * all other types cannot be promoted to double, therfore
     * will return ill_formed_type.
     *
     * @param et expression type
     * @return promoted expression type
     */
    bare_expr_type promote_primitive(const bare_expr_type& et);

    /**
     * For args pair (int_type, double_type), returns double_type;
     * pair (int_type, int_type) returns int_type (no promotion),
     * pair (double_type, double_type) returns double_type (no promotion),
     * for all other combinations, return ill_formed_type.
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
