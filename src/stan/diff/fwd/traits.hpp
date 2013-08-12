#ifndef __STAN__AGRAD__FVAR__TRAITS_HPP__
#define __STAN__AGRAD__FVAR__TRAITS_HPP__

#include <stan/meta/traits.hpp>
#include <stan/agrad/fwd/fvar.hpp>

namespace stan {

  template <typename T>
  struct is_fvar<stan::agrad::fvar<T> > {
    enum { value = true };
  };

}

#endif
