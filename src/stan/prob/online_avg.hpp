#ifndef __STAN__PROB__ONLINE_AVG_HPP__
#define __STAN__PROB__ONLINE_AVG_HPP__

#include <vector>
#include <stdexcept>

namespace stan {

  namespace prob {

    class online_avg {
    public:
      online_avg()    
	: _num_samples(0),
	  _mM(0,0.0),
	  _mS(0,0.0) {
      }
   
      online_avg(int N)
	: _num_samples(0),
	  _mM(N,0.0),
	  _mS(N,0.0) {
      }

      ~online_avg() {
      }

      unsigned int num_dimensions() {
	return _mM.size();
      };
      
      /**
       *
       * @throw std::runtime_error if x.size() != num_dimensions()
       */
      void add(std::vector<double>& x) {
	if (x.size() != num_dimensions())
	  throw std::runtime_error("x.size() must equal num_dimensions()");
	++_num_samples;
	for (unsigned int n = 0; n < num_dimensions(); ++n) {
	  double nextM = _mM[n] + (x[n] - _mM[n]) / _num_samples;
	  _mS[n] += (x[n] - _mM[n]) * (x[n] - nextM);
	  _mM[n] = nextM;
	}
      }
      
      /**
       *
       * @throw std::runtime_error if there are no samples
       */
      void remove(std::vector<double>& x) {
	if (num_samples() == 0)
	  throw std::runtime_error ("no samples to remove");
	for (unsigned int n = 0; n < num_dimensions(); ++n) {
	  double m_old = (_num_samples * _mM[n] - x[n])/(_num_samples - 1);
	  _mS[n] -= (x[n] - _mM[n]) * (x[n] - m_old);
	  _mM[n] = m_old;
	}
	--_num_samples;
      }

      unsigned int num_samples() {
	return _num_samples;
      }
      
      /**
       *
       * @throw std::runtime_error if n >= num_dimensions
       */
      double avg(unsigned int n) {
	if (n >= num_dimensions())
	  throw std::runtime_error("n >= num_dimensions()");
	return _mM[n];
      }

      /**
       *
       * @throw std::runtime_error if n >= num_dimensions
       */
      double sample_variance(unsigned int n) {
	if (n >= num_dimensions())
	  throw std::runtime_error("n >= num_dimensions()");
	return _num_samples > 1
	  ? _mS[n] / (_num_samples - 1)
	  : 0.0;
      }

      double sample_deviation(unsigned int n) {
	return sqrt(sample_variance(n));
      }

      /**
       *
       * @throw std::runtime_error if dimension of avgs does not match
       *    num_dimensions()
       */      
      void avgs(std::vector<double>& avgs) {
	if(avgs.size() != num_dimensions())
	  throw std::runtime_error ("avgs.size() must equal num_dimensions()");
	for (unsigned int n = 0; n < num_dimensions(); ++n) 
	  avgs[n] = avg(n);
      }

      /**
       *
       * @throw std::runtime_error if dimension of variances does not match
       *    num_dimensions()
       */      
      void sample_variances(std::vector<double>& variances) {
	if(variances.size() != num_dimensions())
	  throw std::runtime_error ("variances.size() must equal num_dimensions()");
	for (unsigned int n = 0; n < num_dimensions(); ++n) 
	  variances[n] = sample_variance(n);
      }

      /**
       *
       * @throw std::runtime_error if dimension of devs does not match
       *    num_dimensions()
       */      
      void sample_deviations(std::vector<double>& devs) {
	if(devs.size() != num_dimensions())
	  throw std::runtime_error ("devs.size() must equal num_dimensions()");
	for (unsigned int n = 0; n < num_dimensions(); ++n) 
	  devs[n] = sample_deviation(n);
      }

    private:
      int _num_samples;
      std::vector<double> _mM;
      std::vector<double> _mS;

    };

  }

}
#endif
