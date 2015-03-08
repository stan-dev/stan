#ifndef STAN__MATH__PRIM__SCAL__META__ERROR_INDEX_HPP
#define STAN__MATH__PRIM__SCAL__META__ERROR_INDEX_HPP

namespace stan {

  struct error_index {
    enum { value = 
#ifdef ERROR_INDEX
ERROR_INDEX
#else
1
#endif
    };
  };

}
#endif

