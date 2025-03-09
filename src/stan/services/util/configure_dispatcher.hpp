#ifndef STAN_SERVICES_UTIL_CONFIGURE_DISPATCHER_HPP
#define STAN_SERVICES_UTIL_CONFIGURE_DISPATCHER_HPP

#include <stan/callbacks/dispatcher.hpp>
#include <stan/callbacks/json_writer.hpp>
#include <stan/callbacks/unique_stream_writer.hpp>
#include <unordered_map>
#include <memory>
#include <ostream>
#include <string>
#include <sstream>

namespace stan {
namespace services {
namespace util {

/**
 * Creates and configures a dispatcher with appropriate channels based on
 * the provided mapping from info_type to output streams.
 *
 * @param[in] output_streams Map from info_type to shared_ptr<ostream>
 * @return A configured dispatcher object
 */
callbacks::dispatcher configure_dispatcher(
    std::unordered_map<callbacks::info_type, std::shared_ptr<std::ostream>,
                       callbacks::info_type_hash>
        output_streams) {
  callbacks::dispatcher dispatcher;

  for (auto& pair : output_streams) {
    callbacks::info_type type = pair.first;
    std::shared_ptr<std::ostream> stream_ptr = pair.second;

    if (!stream_ptr) {
      std::stringstream ss;
      ss << "Stream for info_type " << static_cast<int>(type) << " is null";
      throw std::runtime_error(ss.str());
    }

    // Create appropriate channel based on info_type
    switch (type) {
      case callbacks::info_type::METRIC: {
        // For METRIC, use a structured_writer_channel with json_writer
        struct deleter_noop {
          void operator()(std::ostream* ptr) const {}
        };

        auto json_writer = std::make_shared<
            callbacks::json_writer<std::ostream, deleter_noop>>(
            std::unique_ptr<std::ostream, deleter_noop>(stream_ptr.get()));

        // Add the writer to the managed resources
        dispatcher.add_managed_resource(json_writer);

        // Create channel using the raw pointer from the shared_ptr
        auto channel = std::make_unique<callbacks::structured_writer_channel>(
            json_writer.get());
        dispatcher.register_channel(type, std::move(channel));
        break;
      }
      case callbacks::info_type::UNCONSTRAINED_INITS:
      case callbacks::info_type::SAMPLE:
      case callbacks::info_type::SAMPLE_RAW:
      case callbacks::info_type::CONFIG:
      case callbacks::info_type::DIAGNOSTIC: {
        // For other types, use a writer_channel with unique_stream_writer
        struct deleter_noop {
          void operator()(std::ostream* ptr) const {}
        };

        auto stream_writer = std::make_shared<
            callbacks::unique_stream_writer<std::ostream, deleter_noop>>(
            std::unique_ptr<std::ostream, deleter_noop>(stream_ptr.get()));

        // Add the writer to the managed resources
        dispatcher.add_managed_resource(stream_writer);

        // Create channel using the raw pointer from the shared_ptr
        auto channel
            = std::make_unique<callbacks::writer_channel>(stream_writer.get());
        dispatcher.register_channel(type, std::move(channel));
        break;
      }
      default:
        std::stringstream ss;
        ss << "Unknown info_type " << static_cast<int>(type)
           << " in configure_dispatcher";
        throw std::runtime_error(ss.str());
    }
  }

  return dispatcher;
}

}  // namespace util
}  // namespace services
}  // namespace stan
#endif
