#ifndef __STAN__PROB__AUTOCORRELATION_HPP__
#define __STAN__PROB__AUTOCORRELATION_HPP__

#include <vector>

#include <kiss_fft/kiss_fftr.h>
#include <vector>
#include <stan/prob/autocorrelation.hpp>

namespace stan {
  
  namespace prob {

    /**
     * Write autocorrelation estimates for every lag for the specified
     * input sequence into the specified result.  The return vector be
     * resized to the same length as the input sequence with lags
     * given by array index. 
     *
     * <p>The implementation involves a fast Fourier transform,
     * followed by a normalization, followed by an inverse transform.
     *
     * @param y Input sequence.
     * @param ac Auto-correlations.
     */
    void auto_correlation(const std::vector<double>& y,
                          std::vector<double>& ac) {

      size_t N = y.size();
      size_t Nt2 = 2 * N;


      // FIXME:  direct malloc prob faster; also z and w below
      std::vector<double> ys_vec(y);
      // zero pad the input to twice its original length
      for (size_t i = 0; i < N; ++i)
        ys_vec.push_back(0.0); 
      double* ys = &ys_vec[0];

      // FIXME: provide scratch buffer for last args; default 0 in fun call
      // allocate working memory and config for FFT
      bool inverse_fft = false;
      kiss_fftr_cfg cfg 
        = kiss_fftr_alloc(Nt2, inverse_fft, 0, 0);

      std::vector<kiss_fft_cpx> z(N + 1);
      kiss_fft_cpx* zs = &z[0];

      // do the forward FFT
      kiss_fftr(cfg,ys,zs);

      // square each value in the complex output
      for (size_t n = 0; n <= N; ++n) {
        zs[n].r = (zs[n].r * zs[n].r) + (zs[n].i * zs[n].i);
        zs[n].i = 0.0;
      }

      free(cfg);

      // FIXME:  re-use config with scratch?
      inverse_fft = true;
      kiss_fftr_cfg cfgi = kiss_fftr_alloc(Nt2, inverse_fft, 0, 0);

      std::vector<double> w(2 * N);
      double* ws = &w[0];

      // inverse transform results in auto-correlations indexed by lag
      kiss_fftri(cfgi,zs,ws);

      // write into output vector
      ac.resize(N);
      double N_over_ws0 = N / ws[0];
      for (size_t n = 0; n < N; ++n)
        ac[n] = ws[n] * N_over_ws0 / (N - n);
      // ac[n] = ((N / (N - n)) * ws[n]) / ws[0];
      free(cfgi);
    }


  }
}

#endif
