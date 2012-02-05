#ifndef __STAN__MEMORY__STACK_ALLOC_HPP__
#define __STAN__MEMORY__STACK_ALLOC_HPP__

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <stdint.h> // FIXME: replace with cstddef?
#include <vector>

namespace stan { 

  namespace memory { 

    /**
     * Return <code>true</code> if the specified pointer is aligned
     * on the number of bytes.
     *
     * This doesn't really make sense other than for powers of 2.
     *
     * @param ptr Pointer to test.
     * @param bytes_aligned Number of bytes of alignment required.
     * @return <code>true</code> if pointer is aligned.
     * @tparam Type of object to which pointer points.
     */
    template <typename T>
    bool is_aligned(T* ptr, unsigned int bytes_aligned) {
      return (reinterpret_cast<uintptr_t>(ptr) % bytes_aligned) == 0U;
    }


    namespace {
      const size_t DEFAULT_INITIAL_NBYTES = 1 << 16; // 64KB


      // FIXME: enforce alignment
      // big fun to inline, but only called twice
      inline char* eight_byte_aligned_malloc(size_t size) {
        char* ptr = static_cast<char*>(malloc(size));
        if (!ptr) return ptr; // malloc failed to alloc
        if (!is_aligned(ptr,8U)) {
          std::stringstream s;
          s << "invalid alignment to 8 bytes, ptr=" 
            << reinterpret_cast<uintptr_t>(ptr) 
            << std::endl;
          throw std::runtime_error(s.str());
        }
        return ptr;
      }
    }

    
    /**
     * Here's an example of how to use HMC for a simple model with
     * strong parameter correlations.
     *
     * @example bivar_norm.cpp  Simple example of full HMC sampling.
     */


    /**
     * An instance of this class provides a memory pool through
     * which blocks of raw memory may be allocated and then collected
     * simultaneously.
     * 
     * This class is useful in settings where large numbers of small
     * objects are allocated and then collected all at once.  This may
     * include objects whose destructors have no effect.
     * 
     * Memory is allocated on a stack of blocks.  Each block allocated
     * is twice as large as the previous one.  The memory may be
     * recovered, with the blocks being reused, or all blocks may be
     * freed, resetting the stack of blocks to its original state. 
     *
     * Alignment up to 8 byte boundaries guaranteed for the first malloc,
     * and after that it's up to the caller.  On 64-bit architectures,
     * all struct values should be padded to 8-byte boundaries if they
     * contain an 8-byte member or a virtual function.
     */
    class stack_alloc {
    private: 
      std::vector<char*> blocks_; // storage for blocks, may be bigger than cur_block_
      std::vector<size_t> sizes_; // could store initial & shift for others
      unsigned int cur_block_;    // index into blocks_ for next alloc
      size_t used_;               // how much of current block already used
    public:


      /**
       * Construct a resizable stack allocator initially holding the
       * specified number of bytes.
       *
       * @param $initial_nbytes Initial number of bytes for the
       * allocator.  Defaults to <code>(1 << 16) = 64KB</code> initial bytes.
       * @throws std::runtime_error if the underlying malloc is not 8-byte
       * aligned.
       */
      stack_alloc(size_t initial_nbytes = DEFAULT_INITIAL_NBYTES) :
        blocks_(1, eight_byte_aligned_malloc(initial_nbytes)),
        sizes_(1,initial_nbytes),
        cur_block_(0),
        used_(0) {

        if (!blocks_[0])
          throw std::bad_alloc();  // no msg allowed in bad_alloc ctor
      }

      /**
       * Destroy this memory allocator.
       *
       * This is implemented as a no-op as there is no destruction
       * required.
       */
      ~stack_alloc() { 
        // free ALL blocks
        for (unsigned int i = 0; i < blocks_.size(); ++i)
          if (blocks_[i])
            free(blocks_[i]);
      }

      /**
       * Return a newly allocated block of memory of the appropriate
       * size managed by the stack allocator.
       *
       * The allocated pointer will be 8-byte aligned.
       *
       * This function may call C++'s <code>malloc()</code> function,
       * with any exceptions percolated throught this function.
       *
       * @param size_t $len Number of bytes to allocate.
       * @return A pointer to the allocated memory.
       */
      inline void* alloc(size_t len) {
        // not enough space in current block
        if (sizes_[cur_block_] < used_ + len) {
          ++cur_block_;
          used_ = 0; // not using anything in next blocks
        }

        // continue skipping blocks that are too small
        while (cur_block_ < blocks_.size() && sizes_[cur_block_] < len)
          ++cur_block_;
                   
        // alloc block if necessary
        if (cur_block_ >= sizes_.size()) {
          // malloc if can't reuse
          size_t newsize = sizes_.back() * 2;
          if (newsize < len) // could keep doubling until big enough
            newsize = len;
          char* bytes = eight_byte_aligned_malloc(newsize);
          if (!bytes)
            throw std::bad_alloc(); // no msg allowed in bad_alloc ctor
          blocks_.push_back(bytes);
          sizes_.push_back(newsize);
          used_ = 0;
        }

        void* result = &blocks_[cur_block_][used_];
        used_ += len;
        return result;
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
        for (unsigned int i = 1; i < blocks_.size(); ++i)
          if (blocks_[i])
            free(blocks_[i]);
        sizes_.resize(1);
        blocks_.resize(1); 
        recover_all();
      }
  
    };

  }
}
#endif
