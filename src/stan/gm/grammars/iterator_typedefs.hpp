#ifndef STAN__GM__PARSER__PARSER__ITERATOR_TYPEDEFS__HPP
#define STAN__GM__PARSER__PARSER__ITERATOR_TYPEDEFS__HPP

#include <boost/spirit/include/version.hpp>
#include <boost/spirit/home/support/iterators/line_pos_iterator.hpp>

namespace stan {
  namespace gm {
    typedef std::string::const_iterator input_iterator_t;
    typedef boost::spirit::line_pos_iterator<input_iterator_t> pos_iterator_t;
  }
}
#endif
