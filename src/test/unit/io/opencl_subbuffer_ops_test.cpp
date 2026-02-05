#ifdef STAN_OPENCL
#include <stan/io/opencl/utils.hpp>
#include <stan/math/opencl/prim.hpp>
#include <stan/math/opencl/copy.hpp>
#include <test/unit/math/expect_near_rel.hpp>
#include <gtest/gtest.h>

TEST(opencl_subbuffer_ops, add_subbuffers) {
  std::vector<size_t> sizes{4, 4};
  const size_t align_elems = stan::io::internal::align_elems_from_device();
  const auto layout = stan::io::compute_serializer_layout(sizes, align_elems);

  Eigen::VectorXd params(8);
  for (Eigen::Index i = 0; i < params.size(); ++i) {
    params.coeffRef(i) = static_cast<double>(i + 1);
  }

  auto values = stan::io::allocate_serializer_buffer(layout, CL_MEM_READ_ONLY);
  stan::io::copy_to_serialize_buffer(params, values, layout);

  cl::Buffer parent = values.buffer();
  cl_buffer_region region_a{layout.offsets[0] * sizeof(double),
                            sizes[0] * sizeof(double)};
  cl_buffer_region region_b{layout.offsets[1] * sizeof(double),
                            sizes[1] * sizeof(double)};

  cl::Buffer sub_a = parent.createSubBuffer(CL_MEM_READ_ONLY,
                                            CL_BUFFER_CREATE_TYPE_REGION,
                                            &region_a);
  cl::Buffer sub_b = parent.createSubBuffer(CL_MEM_READ_ONLY,
                                            CL_BUFFER_CREATE_TYPE_REGION,
                                            &region_b);

  stan::math::matrix_cl<double> a(sub_a, 2, 2);
  stan::math::matrix_cl<double> b(sub_b, 2, 2);

  stan::math::matrix_cl<double> sum = a + b;
  Eigen::MatrixXd sum_host = stan::math::from_matrix_cl<Eigen::MatrixXd>(sum);

  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::ColMajor>>
      a_ref(params.data(), 2, 2);
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                  Eigen::ColMajor>>
      b_ref(params.data() + 4, 2, 2);

  Eigen::MatrixXd expected = a_ref + b_ref;
  stan::test::expect_near_rel("opencl_subbuffer_ops", sum_host, expected);
}

#else
#include <gtest/gtest.h>
TEST(opencl_subbuffer_ops, dummy) { EXPECT_NO_THROW(); }
#endif
