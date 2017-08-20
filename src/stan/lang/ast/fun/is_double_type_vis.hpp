#ifndef STAN_LANG_AST_FUN_IS_DOUBLE_TYPE_VIS_HPP
#define STAN_LANG_AST_FUN_IS_DOUBLE_TYPE_VIS_HPP

#include <boost/variant/static_visitor.hpp>

namespace stan {
  namespace lang {

    struct ill_formed_type;
    struct void_type;
    struct int_type;
    struct double_type;
    struct vector_type;
    struct row_vector_type;
    struct matrix_type;

    /**
     * Visitor to determine if a base_expr_type is double type.
     */
    struct is_double_type_vis : public boost::static_visitor<bool> {
    
      /**
       * Return true if the specified base expression type is double type.
       *
       * @param base_type base expression type
       * @return false
       */
      bool operator()(const ill_formed_type& base_type) const;
    
      /**
       * Return true if the specified base expression type is double type.
       *
       * @param base_type base expression type
       * @return false
       */
      bool operator()(const void_type& base_type) const;

      /**
       * Return true if the specified base expression type is double type.
       *
       * @param base_type base expression type
       * @return false
       */
      bool operator()(const int_type& base_type) const;

      /**
       * Return true if the specified base expression type is double type.
       *
       * @param base_type base expression type
       * @return true
       */
      bool operator()(const double_type& base_type) const;

      /**
       * Return true if the specified base expression type is double type.
       *
       * @param base_type base expression type
       * @return false
       */
      bool operator()(const vector_type& base_type) const;

      /**
       * Return true if the specified base expression type is double type.
       *
       * @param base_type base expression type
       * @return false
       */
      bool operator()(const row_vector_type& base_type) const;

      /**
       * Return true if the specified base expression type is double type.
       *
       * @param base_type base expression type
       * @return false
       */
      bool operator()(const matrix_type& base_type) const;
    };

  }
}
#endif
