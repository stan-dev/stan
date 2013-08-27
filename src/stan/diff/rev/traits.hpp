#ifndef __STAN__DIFF__REV__TRAITS_HPP__
#define __STAN__DIFF__REV__TRAITS_HPP__

#include <stan/meta/traits.hpp>
#include <stan/diff/rev/var.hpp>

namespace stan {

  template <>
  struct is_var<stan::diff::var> {
    enum { value = true };
  };

}
#endif
