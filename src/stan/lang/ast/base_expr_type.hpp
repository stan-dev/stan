#ifndef STAN_LANG_AST_BASE_EXPR_TYPE_HPP
#define STAN_LANG_AST_BASE_EXPR_TYPE_HPP

namespace stan {
  namespace lang {

    /**
     * The type of a base expression.  This is a typedef rather than
     * an enum to get around forward declaration issues with enums in
     * header files.
     */
    typedef int base_expr_type;

    /**
     * Void type.  Used as return type for void functions.
     */
    const int VOID_T = 0;

    /**
     * Integer type.
     */
    const int INT_T = 1;

    /**
     * Real scalar type.
     */
    const int DOUBLE_T = 2;

    /**
     * Column vector type; scalar type is real.
     */
    const int VECTOR_T = 3;

    /**
     * Row vector type; scalar type is real.
     */
    const int ROW_VECTOR_T = 4;

    /**
     * Matrix type; scalar type is real.
     */
    const int MATRIX_T = 5;

    /**
     * Type denoting an ill-formed expression.  Used as a return for
     * functions. 
     */
    const int ILL_FORMED_T = 6;

  }
}
#endif
