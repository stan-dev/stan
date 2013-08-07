#ifndef __STAN__AGRAD__REV__TRAITS_HPP__
#define __STAN__AGRAD__REV__TRAITS_HPP__

#include <stan/meta/traits.hpp>
#include <stan/agrad/rev/var.hpp>

namespace stan {

  template <>
  struct is_var<stan::agrad::var> {
    enum { value = true };
  };

}
#endif
