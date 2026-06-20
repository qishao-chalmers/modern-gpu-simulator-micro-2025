// Copyright (c) 2009-2011, Tor M. Aamodt
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef MEM_FETCH_H
#define MEM_FETCH_H

#include <bitset>
#include "../abstract_hardware_model.h"
#include "addrdec.h"

enum mf_type {
  READ_REQUEST = 0,
  WRITE_REQUEST,
  READ_REPLY,  // send to shader
  WRITE_ACK
};

#define MF_TUP_BEGIN(X) enum X {
#define MF_TUP(X) X
#define MF_TUP_END(X) \
  }                   \
  ;
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

// Qi: per-stage mem_fetch latency mean/variance, split by whether the
// request's PC falls in a configured "target" range (e.g. the divergence
// segment found via [bar_arrival_trace]) vs everything else.
// Enabled by -mem_fetch_stage_latency_debug 1.
struct mem_fetch_stage_bucket {
  unsigned long long count[NUM_MEM_REQ_STAT];
  unsigned long long sum[NUM_MEM_REQ_STAT];
  double sum_sq[NUM_MEM_REQ_STAT];
};
struct mem_fetch_stage_stats_t {
  mem_fetch_stage_bucket target;  // pc in [pc_lo, pc_hi]
  mem_fetch_stage_bucket other;
};
extern mem_fetch_stage_stats_t g_mem_fetch_stage_stats;
extern bool g_mem_fetch_stage_latency_debug;
extern unsigned long long g_mem_fetch_stage_latency_period;
extern address_type g_mem_fetch_stage_latency_pc_lo;
extern address_type g_mem_fetch_stage_latency_pc_hi;
void mem_fetch_stage_stats_print_and_reset(unsigned long long cycle);

class memory_config;
class mem_fetch {
 public:
  mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
            unsigned ctrl_size, unsigned wid, unsigned sid, unsigned tpc,
            const memory_config *config, unsigned long long cycle,
            mem_fetch *original_mf = NULL, mem_fetch *original_wr_mf = NULL, unsigned int unique_function_id = 0);
  ~mem_fetch();

  void set_status(enum mem_fetch_status status, unsigned long long cycle);
  void set_reply() {
    assert(m_access.get_type() != L1_WRBK_ACC &&
           m_access.get_type() != L2_WRBK_ACC);
    if (m_type == READ_REQUEST) {
      assert(!get_is_write());
      m_type = READ_REPLY;
    } else if (m_type == WRITE_REQUEST) {
      assert(get_is_write());
      m_type = WRITE_ACK;
    }
  }
  void do_atomic();

  void print(FILE *fp, bool print_inst = true) const;

  const addrdec_t &get_tlx_addr() const { return m_raw_addr; }
  void set_chip(unsigned chip_id) { m_raw_addr.chip = chip_id; }
  void set_parition(unsigned sub_partition_id) {
    m_raw_addr.sub_partition = sub_partition_id;
  }
  void set_original_mf(mem_fetch * orig_mf) { original_mf = orig_mf; } // MOD. Added L0I
  mem_access_t& get_access() { return m_access; } // MOD. Added L0I
  bool get_is_filling_L0() { return m_is_filling_L0; } // MOD. Added L0I
  void set_is_filling_L0(bool is_filling_L0) { m_is_filling_L0 = is_filling_L0; } // MOD. Added L0I
  int get_subcore() { return m_subcore; } // MOD. Added L0I
  void set_subcore(int subcore) { m_subcore = subcore; } // MOD. Added L0I
  unsigned get_data_size() const { return m_data_size; }
  void set_data_size(unsigned size) { m_data_size = size; }
  unsigned get_ctrl_size() const { return m_ctrl_size; }
  unsigned size() const { return m_data_size + m_ctrl_size; }
  bool is_write() { return m_access.is_write(); }
  void set_addr(new_addr_type addr) { m_access.set_addr(addr); }
  new_addr_type get_addr() const { return m_access.get_addr(); }
  unsigned get_access_size() const { return m_access.get_size(); }
  new_addr_type get_partition_addr() const { return m_partition_addr; }
  unsigned get_sub_partition_id() const { return m_raw_addr.sub_partition; }
  bool get_is_write() const { return m_access.is_write(); }
  unsigned get_request_uid() const { return m_request_uid; }
  unsigned get_sid() const { return m_sid; }
  unsigned get_tpc() const { return m_tpc; }
  unsigned get_wid() const { return m_wid; }
  bool istexture() const;
  bool isconst() const;
  enum mf_type get_type() const { return m_type; }
  void set_type(enum mf_type type) { m_type = type; }
  bool isatomic() const;

  void set_return_timestamp(unsigned t) { m_timestamp2 = t; }
  void set_icnt_receive_time(unsigned t) { m_icnt_receive_time = t; }
  unsigned get_timestamp() const { return m_timestamp; }
  unsigned get_return_timestamp() const { return m_timestamp2; }
  unsigned get_icnt_receive_time() const { return m_icnt_receive_time; }

  // Qi: per-request DRAM dwell time, for [mem_request_trace]. Set only if
  // this request actually misses L2 and enters the DRAM subsystem (latency
  // queue -> real bank/row-buffer timing model -> back to L2); requests that
  // hit L2 never touch these, so m_dram_enter_cycle stays at the sentinel.
  void set_dram_enter_cycle(unsigned long long t) { m_dram_enter_cycle = t; }
  void set_dram_exit_cycle(unsigned long long t) { m_dram_exit_cycle = t; }
  bool went_to_dram() const {
    return m_dram_enter_cycle != (unsigned long long)-1;
  }
  unsigned long long get_dram_enter_cycle() const { return m_dram_enter_cycle; }
  unsigned long long get_dram_exit_cycle() const { return m_dram_exit_cycle; }

  enum mem_access_type get_access_type() const { return m_access.get_type(); }
  const active_mask_t &get_access_warp_mask() const {
    return m_access.get_warp_mask();
  }
  mem_access_byte_mask_t get_access_byte_mask() const {
    return m_access.get_byte_mask();
  }
  mem_access_sector_mask_t get_access_sector_mask() const {
    return m_access.get_sector_mask();
  }

  addr_t get_access_address() const { return m_access.get_addr(); } // MOD. Added L0I

  // Qi: sector-split/internally-generated mem_fetch objects (see
  // memory_sub_partition::push -> breakdown_request_to_sector_requests,
  // l2cache.cc partition_mf_allocator::alloc) are constructed with inst=NULL,
  // so m_inst is empty on them even though they originated from a real
  // instruction. Fall back through the original_mf chain (set up precisely
  // for this kind of request-splitting) so the PC survives past the L2
  // partition boundary instead of reverting to -1.
  address_type get_pc() const {
    if (!m_inst.empty()) return m_inst.pc;
    if (original_mf != NULL) return original_mf->get_pc();
    return (address_type)-1;
  }
  warp_inst_t &get_inst() { return m_inst; } // MOD. VPREG. Removed const
  enum mem_fetch_status get_status() const { return m_status; }

  const memory_config *get_mem_config() { return m_mem_config; }

  unsigned get_num_flits(bool simt_to_mem);

  mem_fetch *get_original_mf() { return original_mf; }
  mem_fetch *get_original_wr_mf() { return original_wr_mf; }

  void set_is_fixed_latency_constant_access(bool is_fixed_latency_constant_access) { m_is_fixed_latency_constant_access = is_fixed_latency_constant_access; }
  bool get_is_fixed_latency_constant_access() { return m_is_fixed_latency_constant_access; }

  unsigned int get_unique_function_id() { return m_unique_function_id; }

  void set_unique_function_id(unsigned int unique_function_id) { m_unique_function_id = unique_function_id; }

  bool get_is_prefetch() { return m_is_prefetch; }

  void set_is_prefetch(bool is_prefetch) { m_is_prefetch = is_prefetch; }

  unsigned int get_stream_buffer_id() { return m_stream_buffer_id; }

  void set_stream_buffer_id(unsigned int stream_buffer_id) { m_stream_buffer_id = stream_buffer_id; }

  void set_kernel_id(unsigned int kernel_id) { m_kernel_id = kernel_id; }
  unsigned int get_kernel_id() { return m_kernel_id; }

  void set_tlb_way_idx(unsigned int tlb_way_idx) { m_tlb_way_idx = tlb_way_idx; }
  int get_tlb_way_idx() { return m_tlb_way_idx; }
  void set_tlb_set_idx(unsigned int tlb_set_idx) { m_tlb_set_idx = tlb_set_idx; }
  int get_tlb_set_idx() { return m_tlb_set_idx; }
  void set_tlb_tag(new_addr_type tlb_tag) { m_tlb_tag = tlb_tag; }
  new_addr_type get_tlb_tag() { return m_tlb_tag; }

 private:
  // request source information
  unsigned m_request_uid;
  unsigned m_sid;
  unsigned m_tpc;
  unsigned m_wid;
  unsigned int m_kernel_id;

  int m_subcore; // MOD. Added L0I
  bool m_is_filling_L0; // MOD. Added L0I
  bool m_is_prefetch;
  unsigned int m_stream_buffer_id;
  bool m_is_fixed_latency_constant_access;
  unsigned int m_unique_function_id;

  // where is this request now?
  enum mem_fetch_status m_status;
  unsigned long long m_status_change;

  // request type, address, size, mask
  mem_access_t m_access;
  unsigned m_data_size;  // how much data is being written
  unsigned
      m_ctrl_size;  // how big would all this meta data be in hardware (does not
                    // necessarily match actual size of mem_fetch)
  new_addr_type
      m_partition_addr;  // linear physical address *within* dram partition
                         // (partition bank select bits squeezed out)
  addrdec_t m_raw_addr;  // raw physical address (i.e., decoded DRAM
                         // chip-row-bank-column address)
  enum mf_type m_type;

  // statistics
  unsigned
      m_timestamp;  // set to gpu_sim_cycle+gpu_tot_sim_cycle at struct creation
  unsigned m_timestamp2;  // set to gpu_sim_cycle+gpu_tot_sim_cycle when pushed
                          // onto icnt to shader; only used for reads
  unsigned m_icnt_receive_time;  // set to gpu_sim_cycle + interconnect_latency
                                 // when fixed icnt latency mode is enabled

  // Qi: per-request DRAM dwell time tracking, see set_dram_enter_cycle/
  // set_dram_exit_cycle/went_to_dram above. Sentinel -1 = never entered DRAM.
  unsigned long long m_dram_enter_cycle = (unsigned long long)-1;
  unsigned long long m_dram_exit_cycle = (unsigned long long)-1;

  // requesting instruction (put last so mem_fetch prints nicer in gdb)
  warp_inst_t m_inst;

  static unsigned sm_next_mf_request_uid;

  const memory_config *m_mem_config;
  unsigned icnt_flit_size;

  int m_tlb_way_idx;
  int m_tlb_set_idx;
  new_addr_type m_tlb_tag;

  mem_fetch
      *original_mf;  // this pointer is set up when a request is divided into
                     // sector requests at L2 cache (if the req size > L2 sector
                     // size), so the pointer refers to the original request
  mem_fetch *original_wr_mf;  // this pointer refers to the original write req,
                              // when fetch-on-write policy is used
};

#endif
