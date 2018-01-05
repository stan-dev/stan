#ifndef STAN_LANG_AST_FUN_VAR_TYPE_HPP
#define STAN_LANG_AST_FUN_VAR_TYPE_HPP

#include <stan/lang/ast/type/bare_expr_type.hpp>

namespace stan {
  namespace lang {

    /** 
     * Fun variable type is a composite of a bare_expr_type 
     * and bool flag `is_data`
     */
    struct fun_var_type {
      /**
       * Variable bare type.
       */
      bare_expr_type bare_type_;

      /**
       * Flags whether is_data_
       */
      bool is_data_;

      /**
       * Construct a default function variable type.
       */
      fun_var_type();

      /**
       * Construct a function variable type from specified type.
       *
       * @param bare_type variable type
       */
      fun_var_type(const bare_expr_type& bare_type);

      /**
       * Construct a function variable type from specified type and flag.
       *
       * @param bare_type variable type
       * @param is_data true if type is qualified data only.
       */
      fun_var_type(const bare_expr_type& bare_type,
                   bool is_data);


      /**
       * Return true if the specified bare type component is the same as
       * this bare type component.
       *
       * @param fvar_type Other bare type component.
       * @return result of equality test.
       */
      bool operator==(const fun_var_type& fvar_type) const;

      /**
       * Return true if the specified bare type component is not the same as
       * this bare type component.
       *
       * @param fvar_type Other bare type component.
       * @return result of inequality test.
       */
      bool operator!=(const fun_var_type& fvar_type) const;

      /**
       * Return true if this bare type component `order_id_` 
       * is less than that of the specified bare type component.
       *
       * @param fvar_type Other bare type component.
       * @return result of comparison.
       */
      bool operator<(const fun_var_type& fvar_type) const;

      /**
       * Return true if this bare type component `order_id_` 
       * is less than or equal to that of the specified bare type component.
       *
       * @param fvar_type Other bare type component.
       * @return result of comparison.
       */
      bool operator<=(const fun_var_type& fvar_type) const;

      /**
       * Return true if this bare type component `order_id_` 
       * is greater than that of the specified bare type component.
       *
       * @param fvar_type Other bare type component.
       * @return result of comparison.
       */
      bool operator>(const fun_var_type& fvar_type) const;

      /**
       * Return true if this bare type component `order_id_` 
       * is greater than or equal to that of the specified bare type component.
       *
       * @param fvar_type Other bare type component.
       * @return result of comparison.
       */
      bool operator>=(const fun_var_type& fvar_type) const;
    };

    /**
     * Stream a user-readable version of the fun_var_decl to the
     * specified output stream, returning the speicifed argument
     * output stream to allow chaining.
     *
     * @param o output stream
     * @param fvar fun_var_decl
     * @return argument output stream
     */
    std::ostream& operator<<(std::ostream& o,
                             const fun_var_type& fv_type);

  }
}
#endif
