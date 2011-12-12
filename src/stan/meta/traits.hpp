#include <boost/type_traits.hpp>

namespace stan {

  template <typename T>
  struct is_constant {
    enum { value = boost::is_convertible<T,double>::value };
  };

}
