#ifndef STAN__MEMORY__CHUNK_ALLOC_HPP
#define STAN__MEMORY__CHUNK_ALLOC_HPP

#include <cstdlib>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#if defined(_MSC_VER)
    #include <msinttypes.h>  // Microsoft Visual Studio lacks compliant stdint.h
#else
    #include <stdint.h> // FIXME: replace with cstddef?
#endif
#include <vector>
#include <stan/memory/stack_alloc.hpp>

#define DEFAULT_INITIAL_NCHUNKS (1 << 8)
 
namespace stan { 
  namespace memory { 
    /**
     */
    template<typename T, size_t Tnchunks_per_block = DEFAULT_INITIAL_NCHUNKS>
    class chunk_alloc {
    private:
      std::vector<char*> blocks_; // storage for blocks, may be bigger than cur_block_
      size_t cur_block_;          // index into blocks_ for next alloc
      size_t used_;          // index into blocks_[cur_block_] for next alloc
    public:
      
      
      /**
       * Construct a resizable chunk allocator initially holding the
       * specified number of bytes.
       *
       * @throws std::runtime_error if the underlying malloc is not 8-byte
       * aligned.
       */
      chunk_alloc() :
      blocks_(1, eight_byte_aligned_malloc(sizeof(T)*Tnchunks_per_block)),
      used_(0)
      {
        if (!blocks_[0])
          throw std::bad_alloc();  // no msg allowed in bad_alloc ctor
      }
      
      /**
       * Destroy this memory allocator.
       *
       * This is implemented as a no-op as there is no destruction
       * required.
       */
      ~chunk_alloc() { 
        // free ALL blocks
        for (size_t i = 0; i < blocks_.size(); ++i)
          if (blocks_[i])
            free(blocks_[i]);
      }
      
      /**
       * Return a newly allocated chunk of memory of the appropriate
       * size managed by the stack allocator.
       *
       * @return A pointer to the allocated memory.
       */
      inline void* alloc() {
        char *result;
        if (unlikely(used_ >= Tnchunks_per_block)) {
          used_ = 0;
          cur_block_++;
        }
        if (unlikely(cur_block_ >= blocks_.size())) {
          result = eight_byte_aligned_malloc(Tnchunks_per_block*sizeof(T));
          if (!result)
            throw std::bad_alloc(); // no msg allowed in bad_alloc ctor
          blocks_.push_back(result);
        }
        result = blocks_[cur_block_] + sizeof(T)*used_;
        ++used_;
        
        return (void*)result;
      }
      
      /**
       * Recover all the memory used by the stack allocator.  The stack
       * of memory blocks allocated so far will be available for further
       * allocations.  To free memory back to the system, use the
       * function free_all().
       */
      inline void recover_all() {
        cur_block_ = 0;
        used_ = 0;
      }
      
      /**
       * Free all memory used by the stack allocator other than the
       * initial block allocation back to the system.  Note:  the
       * destructor will free all memory.
       */
      inline void free_all() {
        // frees all BUT the first (index 0) block
        for (size_t i = 1; i < blocks_.size(); ++i)
          if (blocks_[i])
            free(blocks_[i]);
        blocks_.resize(1); 
        recover_all();
      }
    };
  }
}
#endif
