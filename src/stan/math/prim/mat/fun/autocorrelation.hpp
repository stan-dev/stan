#ifndef STAN_MATH_PRIM_MAT_FUN_AUTOCORRELATION_HPP
#define STAN_MATH_PRIM_MAT_FUN_AUTOCORRELATION_HPP

#include <stan/math/prim/mat/fun/mean.hpp>
#include <unsupported/Eigen/FFT>
#include <complex>
#include <vector>


namespace stan {

  namespace math {

    namespace {
      /**
       * Find the optimal next size for the FFT so that
       * a minimum number of zeros are padded.
       */
      size_t fft_next_good_size(size_t N) {
        if (N <= 2) return 2;
        while (true) {
          size_t m = N;
          while ((m % 2) == 0) m /= 2;
          while ((m % 3) == 0) m /= 3;
          while ((m % 5) == 0) m /= 5;
          if (m <= 1)
            return N;
          N++;
        }
      }
    }

    /**
     * Write autocorrelation estimates for every lag for the specified
     * input sequence into the specified result using the specified
     * FFT engine.  The return vector be resized to the same length as
     * the input sequence with lags given by array index.
     *
     * <p>The implementation involves a fast Fourier transform,
     * followed by a normalization, followed by an inverse transform.
     *
     * <p>An FFT engine can be created for reuse for type double with:
     *
     * <pre>
     *     Eigen::FFT<double> fft;
     * </pre>
     *
     * @tparam T Scalar type.
     * @param y Input sequence.
     * @param ac Autocorrelations.
     * @param fft FFT engine instance.
     */
    template <typename T>
    void autocorrelation(const std::vector<T>& y,
                         std::vector<T>& ac,
                         Eigen::FFT<T>& fft) {
      using std::vector;
      using std::complex;

      size_t N = y.size();
      size_t M = fft_next_good_size(N);
      size_t Mt2 = 2 * M;


      vector<complex<T> > freqvec;

      // centered_signal = y-mean(y) followed by N zeroes
      vector<T> centered_signal(y);
      centered_signal.insert(centered_signal.end(), Mt2-N, 0.0);
      T mean = stan::math::mean(y);
      for (size_t i = 0; i < N; i++)
        centered_signal[i] -= mean;

      fft.fwd(freqvec, centered_signal);
      for (size_t i = 0; i < Mt2; ++i)
        freqvec[i] = complex<T>(norm(freqvec[i]), 0.0);

      fft.inv(ac, freqvec);
      ac.resize(N);

      /*
      vector<T> mask_correction_factors;
      vector<T> mask;
      mask.insert(mask.end(), N, 1.0);
      mask.insert(mask.end(), N, 0.0);

      freqvec.resize(0);
      fft.fwd(freqvec, mask);
      for (size_t i = 0; i < Nt2; ++i)
        freqvec[i] = complex<T>(norm(freqvec[i]), 0.0);

      fft.inv(mask_correction_factors, freqvec);

      for (size_t i = 0; i < N; ++i) {
        ac[i] /= mask_correction_factors[i];
      }
      */
      for (size_t i = 0; i < N; ++i) {
        ac[i] /= (N - i);
      }
      T var = ac[0];
      for (size_t i = 0; i < N; ++i)
        ac[i] /= var;
    }

    /**
     * Write autocorrelation estimates for every lag for the specified
     * input sequence into the specified result.  The return vector be
     * resized to the same length as the input sequence with lags
     * given by array index.
     *
     * <p>The implementation involves a fast Fourier transform,
     * followed by a normalization, followed by an inverse transform.
     *
     * <p>This method is just a light wrapper around the three-argument
     * autocorrelation function
     *
     * @tparam T Scalar type.
     * @param y Input sequence.
     * @param ac Autocorrelations.
     */
    template <typename T>
    void autocorrelation(const std::vector<T>& y,
                         std::vector<T>& ac) {
      Eigen::FFT<T> fft;
      return autocorrelation(y, ac, fft);
    }


  }
}

#endif
