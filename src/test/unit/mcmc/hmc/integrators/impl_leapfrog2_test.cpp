#include <stan/io/dump.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_metric.hpp>
#include <stan/interface_callbacks/writer/stream_writer.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/integrators/impl_leapfrog.hpp>

#include <test/test-models/good/mcmc/hmc/integrators/gauss.hpp>

#include <boost/random/additive_combine.hpp>  // L'Ecuyer RNG

#include <gtest/gtest.h>

#include <sstream>

typedef boost::ecuyer1988 rng_t;

TEST(McmcHmcIntegratorsImplLeapfrog, unit_e_energy_conservation) {
  rng_t base_rng(0);

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream model_output;
  std::stringstream metric_output;
  stan::interface_callbacks::writer::stream_writer writer(metric_output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  gauss_model_namespace::gauss_model model(data_var_context, &model_output);

  stan::mcmc::impl_leapfrog<
    stan::mcmc::unit_e_metric<gauss_model_namespace::gauss_model, rng_t> >
    integrator;

  stan::mcmc::unit_e_metric<gauss_model_namespace::gauss_model, rng_t> metric(model);

  stan::mcmc::unit_e_point z(1);
  z.q(0) = 1;
  z.p(0) = 1;

  metric.init(z, writer, error_writer);
  double H0 = metric.H(z);
  double aveDeltaH = 0;

  double epsilon = 1e-3;
  double tau = 6.28318530717959;
  size_t L = tau / epsilon;

  for (size_t n = 0; n < L; ++n) {
    integrator.evolve(z, metric, epsilon, writer, error_writer);

    double deltaH = metric.H(z) - H0;
    aveDeltaH += (deltaH - aveDeltaH) / double(n + 1);
  }

  // Average error in Hamiltonian should be O(epsilon^{2})
  // in general, smaller for the gauss_model in this case due to cancellations
  EXPECT_NEAR(aveDeltaH, 0, epsilon * epsilon);

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", metric_output.str());
}

TEST(McmcHmcIntegratorsImplLeapfrog, unit_e_symplecticness) {
  rng_t base_rng(0);

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream model_output;
  std::stringstream metric_output;
  stan::interface_callbacks::writer::stream_writer writer(metric_output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  gauss_model_namespace::gauss_model model(data_var_context, &model_output);

  stan::mcmc::impl_leapfrog<
    stan::mcmc::unit_e_metric<gauss_model_namespace::gauss_model, rng_t> >
    integrator;

  stan::mcmc::unit_e_metric<gauss_model_namespace::gauss_model, rng_t> metric(model);

  // Create a circle of points
  const int n_points = 1000;

  double pi = 3.141592653589793;
  double r = 1.5;
  double q0 = 1;
  double p0 = 0;

  std::vector<stan::mcmc::unit_e_point> z;

  for (int i = 0; i < n_points; ++i) {
    z.push_back(stan::mcmc::unit_e_point(1));

    double theta = 2 * pi * (double)i / (double)n_points;
    z.back().q(0) = r * cos(theta) + q0;
    z.back().p(0)    = r * sin(theta) + p0;
  }

  // Evolve circle
  double epsilon = 1e-3;
  size_t L = pi / epsilon;

  for (int i = 0; i < n_points; ++i)
    metric.init(z.at(i), writer, error_writer);

  for (size_t n = 0; n < L; ++n)
    for (int i = 0; i < n_points; ++i)
      integrator.evolve(z.at(i), metric, epsilon, writer, error_writer);

  // Compute area of evolved shape using divergence theorem in 2D
  double area = 0;

  for (int i = 0; i < n_points; ++i) {

    double x1 = z[i].q(0);
    double y1 = z[i].p(0);
    double x2 = z[(i + 1) % n_points].q(0);
    double y2 = z[(i + 1) % n_points].p(0);

    double x_bary = 0.5 * (x1 + x2);
    double y_bary = 0.5 * (y1 + y2);

    double x_delta = x2 - x1;
    double y_delta = y2 - y1;

    double a = sqrt( x_delta * x_delta + y_delta * y_delta);

    double x_norm = 1;
    double y_norm = - x_delta / y_delta;
    double norm = sqrt( x_norm * x_norm + y_norm * y_norm );

    a *= (x_bary * x_norm + y_bary * y_norm) / norm;
    a = a < 0 ? -a : a;

    area += a;

  }

  area *= 0.5;

  // Symplectic integrators preserve volume (area in 2D)
  EXPECT_NEAR(area, pi * r * r, 1e-2);


  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", metric_output.str());
}

TEST(McmcHmcIntegratorsImplLeapfrog, softabs_energy_conservation) {
  rng_t base_rng(0);

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream model_output;
  std::stringstream metric_output;
  stan::interface_callbacks::writer::stream_writer writer(metric_output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  gauss_model_namespace::gauss_model model(data_var_context, &model_output);

  stan::mcmc::impl_leapfrog<
    stan::mcmc::softabs_metric<gauss_model_namespace::gauss_model, rng_t> >
    integrator;

  stan::mcmc::softabs_metric<gauss_model_namespace::gauss_model, rng_t> metric(model);

  stan::mcmc::softabs_point z(1);
  z.q(0) = 1;
  z.p(0) = 1;

  metric.init(z, writer, error_writer);
  double H0 = metric.H(z);
  double aveDeltaH = 0;

  double epsilon = 1e-3;
  double tau = 6.28318530717959;
  size_t L = tau / epsilon;

  for (size_t n = 0; n < L; ++n) {
    integrator.evolve(z, metric, epsilon, writer, error_writer);

    double deltaH = metric.H(z) - H0;
    aveDeltaH += (deltaH - aveDeltaH) / double(n + 1);
  }

  // Average error in Hamiltonian should be O(epsilon^{2})
  // in general, smaller for the gauss_model in this case due to cancellations
  EXPECT_NEAR(aveDeltaH, 0, epsilon * epsilon);

  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", metric_output.str());
}

TEST(McmcHmcIntegratorsImplLeapfrog, softabs_symplecticness) {
  rng_t base_rng(0);

  std::fstream data_stream(std::string("").c_str(), std::fstream::in);
  stan::io::dump data_var_context(data_stream);
  data_stream.close();

  std::stringstream model_output;
  std::stringstream metric_output;
  stan::interface_callbacks::writer::stream_writer writer(metric_output);
  std::stringstream error_stream;
  stan::interface_callbacks::writer::stream_writer error_writer(error_stream);

  gauss_model_namespace::gauss_model model(data_var_context, &model_output);

  stan::mcmc::impl_leapfrog<
    stan::mcmc::softabs_metric<gauss_model_namespace::gauss_model, rng_t> >
    integrator;

  stan::mcmc::softabs_metric<gauss_model_namespace::gauss_model, rng_t> metric(model);

  // Create a circle of points
  const int n_points = 1000;

  double pi = 3.141592653589793;
  double r = 1.5;
  double q0 = 1;
  double p0 = 0;

  std::vector<stan::mcmc::softabs_point> z;

  for (int i = 0; i < n_points; ++i) {
    z.push_back(stan::mcmc::softabs_point(1));

    double theta = 2 * pi * (double)i / (double)n_points;
    z.back().q(0) = r * cos(theta) + q0;
    z.back().p(0)    = r * sin(theta) + p0;
  }

  // Evolve circle
  double epsilon = 1e-3;
  size_t L = pi / epsilon;

  for (int i = 0; i < n_points; ++i)
    metric.init(z.at(i), writer, error_writer);

  for (size_t n = 0; n < L; ++n)
    for (int i = 0; i < n_points; ++i)
      integrator.evolve(z.at(i), metric, epsilon, writer, error_writer);

  // Compute area of evolved shape using divergence theorem in 2D
  double area = 0;

  for (int i = 0; i < n_points; ++i) {

    double x1 = z[i].q(0);
    double y1 = z[i].p(0);
    double x2 = z[(i + 1) % n_points].q(0);
    double y2 = z[(i + 1) % n_points].p(0);

    double x_bary = 0.5 * (x1 + x2);
    double y_bary = 0.5 * (y1 + y2);

    double x_delta = x2 - x1;
    double y_delta = y2 - y1;

    double a = sqrt( x_delta * x_delta + y_delta * y_delta);

    double x_norm = 1;
    double y_norm = - x_delta / y_delta;
    double norm = sqrt( x_norm * x_norm + y_norm * y_norm );

    a *= (x_bary * x_norm + y_bary * y_norm) / norm;
    a = a < 0 ? -a : a;

    area += a;

  }

  area *= 0.5;

  // Symplectic integrators preserve volume (area in 2D)
  EXPECT_NEAR(area, pi * r * r, 1e-2);


  EXPECT_EQ("", model_output.str());
  EXPECT_EQ("", metric_output.str());
}
