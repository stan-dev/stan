#ifndef STAN_ANALYZE_MCMC_AUTOCOVARIANCE_HPP
#define STAN_ANALYZE_MCMC_AUTOCOVARIANCE_HPP

#include <stan/math/prim.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <unsupported/Eigen/FFT>
#include <complex>
#include <vector>

namespace stan {
namespace analyze {

/**
 * Write autocorrelation estimates for every lag for the specified
 * input sequence into the specified result using the specified FFT
 * engine. Normalizes lag-k autocorrelation estimators by N instead
 * of (N - k), yielding biased but more stable estimators as
 * discussed in Geyer (1992); see
 * https://projecteuclid.org/euclid.ss/1177011137. The return vector
 * will be resized to the same length as the input sequence with
 * lags given by array index.
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
template <typename T, typename DerivedA, typename DerivedB>
void autocorrelation(const Eigen::MatrixBase<DerivedA>& y,
                     Eigen::MatrixBase<DerivedB>& ac, Eigen::FFT<T>& fft) {
  size_t N = y.size();
  size_t M = math::internal::fft_next_good_size(N);
  size_t Mt2 = 2 * M;

  // centered_signal = y-mean(y) followed by N zeros
  Eigen::Matrix<T, Eigen::Dynamic, 1> centered_signal(Mt2);
  centered_signal.setZero();
  centered_signal.head(N) = y.array() - y.mean();

  Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> freqvec(Mt2);
  fft.fwd(freqvec, centered_signal);
  // cwiseAbs2 == norm
  freqvec = freqvec.cwiseAbs2();

  Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> ac_tmp(Mt2);
  fft.inv(ac_tmp, freqvec);

  // use "biased" estimate as recommended by Geyer (1992)
  ac = ac_tmp.head(N).real().array() / (N * N * 2);
  ac /= ac(0);
}

/**
 * Write autocovariance estimates for every lag for the specified
 * input sequence into the specified result using the specified FFT
 * engine. Normalizes lag-k autocovariance estimators by N instead
 * of (N - k), yielding biased but more stable estimators as
 * discussed in Geyer (1992); see
 * https://projecteuclid.org/euclid.ss/1177011137. The return vector
 * will be resized to the same length as the input sequence with
 * lags given by array index.
 *
 * <p>The implementation involves a fast Fourier transform,
 * followed by a normalization, followed by an inverse transform.
 *
 * <p>This method is just a light wrapper around the three-argument
 * autocovariance function
 *
 * @tparam T Scalar type.
 * @param y Input sequence.
 * @param acov Autocovariances.
 */
template <typename T, typename DerivedA, typename DerivedB>
void autocovariance(const Eigen::MatrixBase<DerivedA>& y,
                    Eigen::MatrixBase<DerivedB>& acov) {
  Eigen::FFT<T> fft;
  autocorrelation(y, acov, fft);

  using boost::accumulators::accumulator_set;
  using boost::accumulators::stats;
  using boost::accumulators::tag::variance;

  accumulator_set<double, stats<variance>> acc;
  for (int n = 0; n < y.size(); ++n) {
    acc(y(n));
  }

  acov = acov.array() * boost::accumulators::variance(acc);
}

/**
 * Write autocovariance estimates for every lag for the specified
 * input sequence into the specified result using the specified FFT
 * engine. Normalizes lag-k autocovariance estimators by N instead
 * of (N - k), yielding biased but more stable estimators as
 * discussed in Geyer (1992); see
 * https://projecteuclid.org/euclid.ss/1177011137. The return vector
 * will be resized to the same length as the input sequence with
 * lags given by array index.
 *
 * <p>The implementation involves a fast Fourier transform,
 * followed by a normalization, followed by an inverse transform.
 *
 * <p>This method is just a light wrapper around the three-argument
 * autocovariance function
 *
 * @tparam T Scalar type.
 * @param y Input sequence.
 * @param acov Autocovariances.
 */
template <typename T>
void autocovariance(const std::vector<T>& y, std::vector<T>& acov) {
  size_t N = y.size();
  acov.resize(N);

  const Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> y_map(&y[0], N);
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> acov_map(&acov[0], N);
  autocovariance<T>(y_map, acov_map);
}

}  // namespace analyze
}  // namespace stan

#endif
