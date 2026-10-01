#ifndef STAN_IO_OPENCL_UTILS_HPP
#define STAN_IO_OPENCL_UTILS_HPP

#ifdef STAN_OPENCL

#include <stan/math/opencl/opencl_context.hpp>
#include <stan/math/opencl/matrix_cl.hpp>
#include <stan/math/opencl/err/check_opencl.hpp>
#include <stan/math/opencl/rev/vari.hpp>
#include <stan/math/rev/core/var.hpp>
#include <stan/math/prim/err/check_size_match.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/fun/Eigen.hpp>

#include <CL/opencl.hpp>
#include <Eigen/Dense>

#include <numeric>
#include <vector>
#include <utility>
#include <algorithm>

namespace stan {
namespace io {

/**
 * Layout information for aligned OpenCL serializer buffers.
 *
 * Offsets and sizes are in elements (not bytes).
 */
struct serializer_layout {
  /** Size and aligned offset in elements for each block, in read order. */
  std::vector<std::pair<size_t, size_t>> sizes_offsets_;
  /** Total size (including padding) in elements. */
  size_t total_size_{0};
  /** Alignment in elements. */
  size_t align_elems_{1};

  serializer_layout() = default;

  /**
   * Construct aligned offsets and total size for parameter blocks.
   *
   * @param sizes Block sizes in read order (elements).
   * @param align_elems Alignment in elements.
   */
  serializer_layout(const std::vector<size_t>& sizes, size_t align_elems);
};

namespace internal {
/**
 * Round a value up to the next multiple.
 *
 * @param value Value to round up.
 * @param multiple Alignment multiple.
 * @return Rounded value (>= value).
 */
inline size_t round_up(size_t value, size_t multiple) {
  if (multiple == 0) {
    return value;
  }
  const size_t rem = value % multiple;
  return rem == 0 ? value : value + (multiple - rem);
}

/**
 * Query the device alignment and return required alignment in elements.
 *
 * @return Alignment in elements for double buffers.
 * @throws cl::Error if OpenCL device queries fail.
 */
inline size_t align_elems_from_device() {
  size_t align_bits = stan::math::opencl_context.device()[0]
                          .getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
  size_t align_bytes = (align_bits + 7) / 8;
  if (align_bytes == 0) {
    return 1;
  }
  const size_t elem_size = sizeof(double);
  const size_t g = std::gcd(align_bytes, elem_size);
  const size_t align_elems = align_bytes / (g == 0 ? 1 : g);
  return align_elems == 0 ? 1 : align_elems;
}
}  // namespace internal

inline serializer_layout::serializer_layout(const std::vector<size_t>& sizes,
                                            size_t align_elems)
    : align_elems_(std::max<size_t>(1, align_elems)) {
  sizes_offsets_.reserve(sizes.size());
  size_t pos = 0;
  for (size_t size : sizes) {
    pos = internal::round_up(pos, align_elems_);
    sizes_offsets_.emplace_back(size, pos);
    pos += size;
  }
  total_size_ = pos;
}

/**
 * Allocate a matrix_cl buffer sized to the serialized layout.
 *
 * @param total_size Total buffer size in elements, including padding.
 * @param flags OpenCL memory flags.
 * @return Device buffer with shape (total_size, 1).
 * @throws std::system_error if an OpenCL error occurs.
 */
inline stan::math::matrix_cl<double> allocate_serializer_buffer(
    const size_t& total_size, cl_mem_flags flags) {
  if (total_size == 0) {
    return stan::math::matrix_cl<double>();
  }
  cl::Context& ctx = stan::math::opencl_context.context();
  try {
    cl_mem_flags alloc_flags = flags;
    if (stan::math::opencl_context.device()[0]
            .getInfo<CL_DEVICE_HOST_UNIFIED_MEMORY>()) {
      alloc_flags |= CL_MEM_ALLOC_HOST_PTR;
    }
    cl::Buffer buffer(ctx, alloc_flags, sizeof(double) * total_size);
    return stan::math::matrix_cl<double>(buffer, total_size, 1);
  } catch (const cl::Error& e) {
    stan::math::check_opencl_error("allocate_serializer_buffer", e);
  }
  return stan::math::matrix_cl<double>();
}

/**
 * Copy host parameters into aligned blocks of a device buffer.
 *
 * @param src Flat parameters with no padding.
 * @param dst Device buffer to receive padded blocks.
 * @param layout Serializer layout describing offsets and sizes.
 * @throws std::invalid_argument if src size does not match sum of sizes.
 * @throws std::system_error if an OpenCL error occurs.
 */
inline void copy_to_serialize_buffer(const Eigen::VectorXd& src,
                                     stan::math::matrix_cl<double>& dst,
                                     const serializer_layout& layout) {
  const size_t total_src = static_cast<size_t>(src.size());
  const size_t total_sizes = std::accumulate(
      layout.sizes_offsets_.begin(), layout.sizes_offsets_.end(), size_t{0},
      [](size_t total, const auto& size_offset) {
        return total + size_offset.first;
      });
  stan::math::check_size_match("copy_to_serialize_buffer", "src.size()",
                               total_src, "sum(sizes)", total_sizes);
  if (layout.total_size_ == 0) {
    return;
  }

  auto& queue = stan::math::opencl_context.queue();
  std::vector<cl::Event> events;
  events.reserve(layout.sizes_offsets_.size());
  size_t src_offset = 0;

  try {
    for (const auto& [block_size, offset] : layout.sizes_offsets_) {
      if (block_size == 0) {
        continue;
      }
      const size_t origin_bytes = offset * sizeof(double);
      const size_t size_bytes = block_size * sizeof(double);
      cl_buffer_region region{origin_bytes, size_bytes};
      cl::Buffer sub_dst = dst.buffer().createSubBuffer(
          CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &region);

      cl::Event event;
      queue.enqueueWriteBuffer(sub_dst, CL_FALSE, 0, size_bytes,
                               src.data() + src_offset, nullptr, &event);
      events.push_back(event);
      src_offset += block_size;
    }

    for (cl::Event& e : events) {
      e.wait();
    }
  } catch (const cl::Error& e) {
    stan::math::check_opencl_error("copy_to_serialize_buffer", e);
  }
}

/**
 * Serialize host parameters into a padded OpenCL buffer.
 *
 * @param params Flat unconstrained parameters.
 * @param sizes Unconstrained block sizes.
 * @return var_value holding values and adjoints buffers.
 * @throws std::system_error if an OpenCL error occurs.
 * @throws std::invalid_argument if params size does not match sum of sizes.
 */
inline stan::math::var_value<stan::math::matrix_cl<double>> serialize_to_opencl(
    const Eigen::VectorXd& params, const std::vector<size_t>& sizes) {
  const size_t align_elems = internal::align_elems_from_device();
  const serializer_layout layout(sizes, align_elems);

  stan::math::matrix_cl<double> values
      = allocate_serializer_buffer(layout.total_size_, CL_MEM_READ_ONLY);
  stan::math::matrix_cl<double> adjoints
      = allocate_serializer_buffer(layout.total_size_, CL_MEM_READ_WRITE);

  if (layout.total_size_ > 0) {
    auto& queue = stan::math::opencl_context.queue();
    try {
      double zero = 0.0;
      queue.enqueueFillBuffer(values.buffer(), zero, 0,
                              sizeof(double) * layout.total_size_);
      queue.enqueueFillBuffer(adjoints.buffer(), zero, 0,
                              sizeof(double) * layout.total_size_);
      queue.finish();
    } catch (const cl::Error& e) {
      stan::math::check_opencl_error("serialize_to_opencl", e);
    }
  }

  copy_to_serialize_buffer(params, values, layout);

  auto* vi = new stan::math::vari_value<stan::math::matrix_cl<double>>(
      std::move(values), std::move(adjoints));
  return stan::math::var_value<stan::math::matrix_cl<double>>(vi);
}

}  // namespace io
}  // namespace stan

#endif  // STAN_OPENCL

#endif  // STAN_IO_OPENCL_UTILS_HPP
