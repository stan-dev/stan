#ifndef __STAN__DIFF__FVAR__TRAITS_HPP__
#define __STAN__DIFF__FVAR__TRAITS_HPP__

#include <stan/meta/traits.hpp>
#include <stan/diff/fwd/fvar.hpp>

namespace stan {

  template <typename T>
  struct is_fvar<stan::diff::fvar<T> > {
    enum { value = true };
  };

}

#endif
