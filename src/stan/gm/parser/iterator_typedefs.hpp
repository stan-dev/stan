#ifndef __STAN__GM__PARSER__PARSER__ITERATOR_TYPEDEFS__HPP__
#define __STAN__GM__PARSER__PARSER__ITERATOR_TYPEDEFS__HPP__

#include <boost/spirit/include/support_multi_pass.hpp>
#include <boost/spirit/include/classic_position_iterator.hpp>
#include <iterator>

namespace stan {
  namespace gm {
    typedef std::istreambuf_iterator<char> base_iterator_t;
    typedef boost::spirit::multi_pass<base_iterator_t>  forward_iterator_t;
    typedef boost::spirit::classic::position_iterator2<forward_iterator_t> pos_iterator_t;
  }
}
#endif
