#ifndef __STAN__META__TRAITS_HPP__
#define __STAN__META__TRAITS_HPP__

#include <boost/type_traits.hpp>

namespace stan {

  template <typename T>
  struct is_constant {
    enum { value = boost::is_convertible<T,double>::value };
  };

}

#endif
