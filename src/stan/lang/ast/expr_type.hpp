#ifndef STAN_LANG_AST_EXPR_TYPE_HPP
#define STAN_LANG_AST_EXPR_TYPE_HPP

#include <stan/lang/ast/base_expr_type.hpp>
#include <cstddef>

namespace stan {
  namespace lang {

    /**
     * Structure of the type of an expression, which consists of a
     * base type and a number of dimensions.
     */
    struct expr_type {
      /**
       * The base expression type.
       */
      base_expr_type base_type_;

      /**
       * The number of array dimensions.
       */
      std::size_t num_dims_;

      /**
       * Construct an empty expression type.
       */
      expr_type();

      /**
       * Construct an expression type with the specified base type and
       * zero array dimensions.
       *
       * @param base_type base type
       */
      expr_type(const base_expr_type base_type);  // NOLINT(runtime/explicit)

      /**
       * Construct an expression type with the specified base type and
       * specified number of array dimensions.
       *
       * @param base_type base type
       * @param num_dims number of dimensions
       */
      expr_type(const base_expr_type base_type,
                std::size_t num_dims);

      /**
       * Return true if the specified expression type is equal to this
       * expression type in the sense of having the same base type and
       * same number of dimensions.
       *
       * @param et Other expression type.
       * @return result of equality test.
       */
      bool operator==(const expr_type& et) const;

      /**
       * Return true if this expression type is not equal to the
       * specified expression type. 
       *
       * @param et Other expression type.
       * @return result of equality test.
       */
      bool operator!=(const expr_type& et) const;

      /**
       * Return true if this expression type is less than the
       * specified expression type.
       * 
       * <p>Types are ordered lexicographically by the
       * integer value of their base type and then their number of
       * dimensions. 
       *
       * @param et Other expression type.
       * @return result of inequality test.
       */
      bool operator<(const expr_type& et) const;

      /**
       * Return true if this expression type is less than or equal to
       * the specified expression type. 
       *
       * @param et Other expression type.
       * @return result of inequality test.
       */
      bool operator<=(const expr_type& et) const;

      /**
       * Return true if this expression type is greater than the
       * specified expression type.
       *
       * @param et Other expression type.
       * @return result of inequality test.
       */
      bool operator>(const expr_type& et) const;

      /**
       * Return true if this expression type is greater than or equal
       * to the specified expression type.
       *
       * @param et Other expression type.
       * @return result of inequality test.
       */
      bool operator>=(const expr_type& et) const;

      /**
       * Return true if this expression type is an integer or real
       * type with zero dimensions.
       *
       * @return true if this expression type is an integer or real
       * with zero dimensions
       */
      bool is_primitive() const;

      /**
       * Return true if this expression type is an integer type with
       * zero dimensions.
       *
       * @return true if this expression type is an integer with zero
       * dimensions
       */
      bool is_primitive_int() const;

      /**
       * Return true if this expression type is a real type with
       * zero dimensions.
       *
       * @return true if this expression type is a real with zero
       * dimensions
       */
      bool is_primitive_double() const;

      /**
       * Return true if the base type of this type is ill formed.
       *
       * @return bool if this type is ill formed
       */
      bool is_ill_formed() const;

      /**
       * Return true if this type is void.
       *
       * @return bool if this type is void
       */
      bool is_void() const;

      /**
       * Return the base type of this expression type.
       *
       * @return base type of this type
       */
      base_expr_type type() const;

      /**
       * Return the number of dimensions for this type.
       *
       * @return number of dimensions for this type
       */
      std::size_t num_dims() const;
    };

  }
}
#endif
