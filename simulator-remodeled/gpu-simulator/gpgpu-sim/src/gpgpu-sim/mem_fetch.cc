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

#include "mem_fetch.h"
#include "gpu-sim.h"
#include "mem_latency_stat.h"
#include "shader.h"
#include "visualizer.h"
#include <cmath>
#include <algorithm>

unsigned mem_fetch::sm_next_mf_request_uid = 1;

// Qi: per-stage mem_fetch latency mean/variance instrumentation (see mem_fetch.h)
mem_fetch_stage_stats_t g_mem_fetch_stage_stats;
bool g_mem_fetch_stage_latency_debug = false;
unsigned long long g_mem_fetch_stage_latency_period = 10000;
address_type g_mem_fetch_stage_latency_pc_lo = 0;
address_type g_mem_fetch_stage_latency_pc_hi = 0;

mem_fetch::mem_fetch(const mem_access_t &access, const warp_inst_t *inst,
                     unsigned ctrl_size, unsigned wid, unsigned sid,
                     unsigned tpc, const memory_config *config,
                     unsigned long long cycle, mem_fetch *m_original_mf,
                     mem_fetch *m_original_wr_mf, unsigned int unique_function_id)
    : m_access(access)

{
  m_request_uid = sm_next_mf_request_uid++;
  // std::cerr << "Creating mem_fetch: " << m_request_uid << std::endl; fflush(stdout);
  m_access = access;
  if (inst) {
    m_inst = *inst;
    assert(wid == m_inst.warp_id());
  }
  m_data_size = access.get_size();
  m_original_data_size = 0;
  m_ctrl_size = ctrl_size;
  m_sid = sid;
  m_tpc = tpc;
  m_wid = wid;
  config->m_address_mapping.addrdec_tlx(access.get_addr(), &m_raw_addr);
  m_partition_addr =
      config->m_address_mapping.partition_address(access.get_addr());
  m_type = m_access.is_write() ? WRITE_REQUEST : READ_REQUEST;
  m_timestamp = cycle;
  m_timestamp2 = 0;
  m_status = MEM_FETCH_INITIALIZED;
  m_status_change = cycle;
  m_mem_config = config;
  icnt_flit_size = config->icnt_flit_size;
  original_mf = m_original_mf;
  original_wr_mf = m_original_wr_mf;
  if (m_original_mf) {
    m_raw_addr.chip = m_original_mf->get_tlx_addr().chip;
    m_raw_addr.sub_partition = m_original_mf->get_tlx_addr().sub_partition;
  }
  m_subcore = -1; // MOD. Added L0I
  m_is_filling_L0 = false; // MOD. Added L0I
  m_is_fixed_latency_constant_access= false;
  m_unique_function_id = unique_function_id;
  m_is_prefetch = false;
  m_stream_buffer_id = std::numeric_limits<unsigned int>::max();
  m_kernel_id = 0;

  m_tlb_set_idx = -1;
  m_tlb_way_idx = -1;
  m_tlb_tag = 0;
}

mem_fetch::~mem_fetch() {
  // std::cerr << "Destroying mem_fetch: " << m_request_uid << std::endl; fflush(stdout);
  m_status = MEM_FETCH_DELETED; 
}

#define MF_TUP_BEGIN(X) static const char *Status_str[] = {
#define MF_TUP(X) #X
#define MF_TUP_END(X) \
  }                   \
  ;
#include "mem_fetch_status.tup"
#undef MF_TUP_BEGIN
#undef MF_TUP
#undef MF_TUP_END

void mem_fetch::print(FILE *fp, bool print_inst) const {
  fprintf(fp, "  mf: uid=%6u, sid%02u:w%02u, part=%u, ", m_request_uid, m_sid,
          m_wid, m_raw_addr.chip);
  m_access.print(fp);
  if ((unsigned)m_status < NUM_MEM_REQ_STAT)
    fprintf(fp, " status = %s (%llu), ", Status_str[m_status], m_status_change);
  else
    fprintf(fp, " status = %u??? (%llu), ", m_status, m_status_change);
  if (!m_inst.empty() && print_inst)
    m_inst.print(fp);
  else
    fprintf(fp, "\n");
}

void mem_fetch::set_status(enum mem_fetch_status status,
                           unsigned long long cycle) {
  // Qi: record how long this request just spent in the OLD status before
  // overwriting it, bucketed by whether its PC is in the configured target
  // range (-mem_fetch_stage_latency_pc_lo/_hi).
  if (g_mem_fetch_stage_latency_debug && (unsigned)m_status < NUM_MEM_REQ_STAT &&
      cycle >= m_status_change) {
    unsigned long long duration = cycle - m_status_change;
    address_type pc = get_pc();
    bool in_target = (g_mem_fetch_stage_latency_pc_hi > 0) &&
                      (pc >= g_mem_fetch_stage_latency_pc_lo) &&
                      (pc <= g_mem_fetch_stage_latency_pc_hi);
    mem_fetch_stage_bucket &b =
        in_target ? g_mem_fetch_stage_stats.target : g_mem_fetch_stage_stats.other;
    b.count[m_status]++;
    b.sum[m_status] += duration;
    b.sum_sq[m_status] += (double)duration * (double)duration;
  }
  m_status = status;
  m_status_change = cycle;
}

static void print_mem_fetch_stage_bucket(const char *label,
                                         const mem_fetch_stage_bucket &b) {
  for (unsigned s = 0; s < NUM_MEM_REQ_STAT; s++) {
    if (b.count[s] == 0) continue;
    double n = (double)b.count[s];
    double mean = (double)b.sum[s] / n;
    double var = b.sum_sq[s] / n - mean * mean;
    if (var < 0.0) var = 0.0;
    printf("[mem_stage_stats] %-7s %-32s n=%-10llu mean=%-10.2f var=%-12.2f stdev=%-8.2f\n",
           label, Status_str[s], b.count[s], mean, var, std::sqrt(var));
  }
}

void mem_fetch_stage_stats_print_and_reset(unsigned long long cycle) {
  printf("[mem_stage_stats] ==== cycle=%llu pc_range=[0x%llx,0x%llx] ====\n", cycle,
         (unsigned long long)g_mem_fetch_stage_latency_pc_lo,
         (unsigned long long)g_mem_fetch_stage_latency_pc_hi);
  print_mem_fetch_stage_bucket("target", g_mem_fetch_stage_stats.target);
  print_mem_fetch_stage_bucket("other", g_mem_fetch_stage_stats.other);
  fflush(stdout);
  memset(&g_mem_fetch_stage_stats, 0, sizeof(g_mem_fetch_stage_stats));
}

bool mem_fetch::isatomic() const {
  if (m_inst.empty()) return false;
  return m_inst.isatomic();
}

void mem_fetch::do_atomic() { m_inst.do_atomic(m_access.get_warp_mask()); }

bool mem_fetch::istexture() const {
  if (m_inst.empty()) return false;
  return m_inst.space.get_type() == tex_space;
}

bool mem_fetch::isconst() const {
  if (m_inst.empty()) return false;
  return (m_inst.space.get_type() == const_space) ||
         (m_inst.space.get_type() == param_space_kernel);
}

/// Returns number of flits traversing interconnect. simt_to_mem specifies the
/// direction
unsigned mem_fetch::get_num_flits(bool simt_to_mem) {
  unsigned sz = 0;
  // If atomic, write going to memory, or read coming back from memory, size =
  // ctrl + data. Else, only ctrl
  if (isatomic() || (simt_to_mem && get_is_write()) ||
      !(simt_to_mem || get_is_write()))
    sz = size();
  else
    sz = get_ctrl_size();

  return (sz / icnt_flit_size) + ((sz % icnt_flit_size) ? 1 : 0);
}
