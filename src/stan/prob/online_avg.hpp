#ifndef __STAN__PROB__ONLINE_AVG_HPP__
#define __STAN__PROB__ONLINE_AVG_HPP__

#include <vector>

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

      void add(std::vector<double>& x) {
	assert(x.size() == num_dimensions());
	++_num_samples;
	for (unsigned int n = 0; n < num_dimensions(); ++n) {
	  double nextM = _mM[n] + (x[n] - _mM[n]) / _num_samples;
	  _mS[n] += (x[n] - _mM[n]) * (x[n] - nextM);
	  _mM[n] = nextM;
	}
      }

      void remove(std::vector<double>& x) {
	assert(_num_samples > 0);
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

      double avg(unsigned int n) {
	assert(n < num_dimensions());
	return _mM[n];
      }

      double sample_variance(unsigned int n) {
	assert(n < num_dimensions());
	return _num_samples > 1
	  ? _mS[n] / (_num_samples - 1)
	  : 0.0;
      }

      double sample_deviation(unsigned int n) {
	return sqrt(sample_variance(n));
      }

      void avgs(std::vector<double>& avgs) {
	assert(avgs.size() == num_dimensions());
	for (unsigned int n = 0; n < num_dimensions(); ++n) 
	  avgs[n] = avg(n);
      }

      void sample_variances(std::vector<double>& variances) {
	assert(variances.size() == num_dimensions());
	for (unsigned int n = 0; n < num_dimensions(); ++n) 
	  variances[n] = sample_variance(n);
      }

      void sample_deviations(std::vector<double>& devs) {
	assert(devs.size() == num_dimensions());
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
