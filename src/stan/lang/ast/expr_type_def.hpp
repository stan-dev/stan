#ifndef STAN_LANG_AST_EXPR_TYPE_DEF_HPP
#define STAN_LANG_AST_EXPR_TYPE_DEF_HPP

#include <stan/lang/ast/ill_formed_type.hpp>
#include <stan/lang/ast/void_type.hpp>
#include <stan/lang/ast/int_type.hpp>
#include <stan/lang/ast/double_type.hpp>
#include <stan/lang/ast/vector_type.hpp>
#include <stan/lang/ast/row_vector_type.hpp>
#include <stan/lang/ast/matrix_type.hpp>
#include <stan/lang/ast/base_expr_type.hpp>

#include <stan/lang/ast/expr_type.hpp>
#include <stan/lang/ast/fun/write_base_expr_type.hpp>
#include <ostream>

namespace stan {
  namespace lang {

    expr_type::expr_type() : base_type_(ill_formed_type()), num_dims_(0) { }

    expr_type::expr_type(const base_expr_type& base_type)
      : base_type_(base_type), num_dims_(0) { }

    expr_type::expr_type(const base_expr_type& base_type,
                         std::size_t num_dims)
      : base_type_(base_type), num_dims_(num_dims) {  }

    bool expr_type::operator==(const expr_type& et) const {
      return base_type_ == et.base_type_ && num_dims_ == et.num_dims_;
    }

    bool expr_type::operator!=(const expr_type& et) const {
      return !(*this == et);
    }

    bool expr_type::operator<(const expr_type& et) const {
      return base_type_ < et.base_type_
             || (base_type_ == et.base_type_ && num_dims_ < et.num_dims_);
    }

    bool expr_type::operator<=(const expr_type& et) const {
      return base_type_ < et.base_type_
             || (base_type_ == et.base_type_ && num_dims_ <= et.num_dims_);
    }

    bool expr_type::operator>(const expr_type& et) const {
      return base_type_ > et.base_type_
             || (base_type_ == et.base_type_ && num_dims_ > et.num_dims_);
    }

    bool expr_type::operator>=(const expr_type& et) const {
      return base_type_ > et.base_type_
             || (base_type_ == et.base_type_ && num_dims_ >= et.num_dims_);
    }

    bool expr_type::is_primitive() const {
      return is_primitive_int()
        || is_primitive_double();
    }

    bool expr_type::is_primitive_int() const {
      return base_type_.is_int_type() && num_dims_ == 0U;
    }

    bool expr_type::is_primitive_double() const {
      return base_type_.is_double_type() && num_dims_ == 0U;
    }

    bool expr_type::is_ill_formed() const {
      return base_type_.is_ill_formed_type();
    }

    bool expr_type::is_void() const {
      return base_type_.is_void_type();
    }

    base_expr_type expr_type::type() const {
      return base_type_;
    }

    size_t expr_type::num_dims() const {
      return num_dims_;
    }

    std::ostream& operator<<(std::ostream& o, const expr_type& et) {
      write_base_expr_type(o, et.type());
      if (et.num_dims() > 0) {
        o << '[';
        for (size_t i = 1; i < et.num_dims(); ++i) o << ",";
        o << ']';
      }
      return o;
    }

  }
}
#endif
