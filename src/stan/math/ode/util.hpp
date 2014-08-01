#ifndef STAN__MATH__ODE__UTIL_HPP__
#define STAN__MATH__ODE__UTIL_HPP__

#include <vector>
#include <stan/agrad/rev/var.hpp>
 
namespace stan {
  namespace agrad {
    stan::agrad::var max(stan::agrad::var a, stan::agrad::var b) {
      return fmax(a, b);
    }
  }
}

namespace stan {
  
  namespace math {

    template<class T>
    struct push_back_state_and_time {
      std::vector< std::vector<T> >& m_states;
      std::vector< T >& m_times;
      
      push_back_state_and_time(std::vector< std::vector<T> > &states,
                               std::vector< T > &times)
        : m_states( states ), 
          m_times( times ) { }
      
      void operator()(const std::vector<T> &x, T t) {
        m_states.push_back( x );
        m_times.push_back( t );
      }
      
      std::vector<std::vector<T> > get() {
        return m_states;
      }
      
      void print() {
        std::cout << "time,x_0";
        for (size_t n = 1; n < m_states[0].size(); n++)
          std::cout << ",x_" << n;
        std::cout << std::endl;
        for (size_t n = 0; n < m_states.size(); n++) {
          std::cout << m_times[n]
                    << "," << m_states[n][0]
                    << "," << m_states[n][1]
                    << std::endl;
        }
      }
    };
    
  }
}
#endif
