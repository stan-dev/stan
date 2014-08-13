#ifndef STAN__GM__PARSER__COMMON_ADAPTORS_DEF__HPP
#define STAN__GM__PARSER__COMMON_ADAPTORS_DEF__HPP

#include <boost/fusion/include/adapt_struct.hpp>
#include <stan/gm/ast.hpp>

#include "boost/fusion/adapted/struct/adapt_struct.hpp"
#include "boost/preprocessor/arithmetic/dec.hpp"
#include "boost/preprocessor/arithmetic/inc.hpp"
#include "boost/preprocessor/control/expr_iif.hpp"
#include "boost/preprocessor/control/iif.hpp"
#include "boost/preprocessor/logical/bool.hpp"
#include "boost/preprocessor/repetition/detail/for.hpp"
#include "boost/preprocessor/seq/elem.hpp"
#include "boost/preprocessor/seq/size.hpp"
#include "boost/preprocessor/tuple/eat.hpp"
#include "boost/preprocessor/tuple/elem.hpp"

BOOST_FUSION_ADAPT_STRUCT(stan::gm::range,
                          (stan::gm::expression, low_)
                          (stan::gm::expression, high_) )


#endif
