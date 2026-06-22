// Copyright (c) 2023-2025, Rodrigo Huerta, Mojtaba Abaie Shoushtary, Josep-Llorenç Cruz, Antonio González
// Universitat Politecnica de Catalunya
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
// The Universitat Politecnica de Catalunya nor the names of its contributors may be
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

// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
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

#include <stdio.h>
#include <stdlib.h>
#include <set>
#include <vector>
#include <unordered_map>
#include <utility>
#include "assert.h"

#ifndef SCOREBOARD_H_
#define SCOREBOARD_H_

#include "../abstract_hardware_model.h"

class Scoreboard {
 public:
  Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t *gpu, bool is_trace_mode);

  void reserveRegisters(const warp_inst_t *inst);
  void reserveRegisters_remodeling(const warp_inst_t *inst);
  void releaseRegisters(const warp_inst_t *inst);
  void releaseRegisters_remodeling(const warp_inst_t *inst);
  void releaseRegister(unsigned wid, unsigned regnum);

  bool checkCollision(unsigned wid, const inst_t *inst) const;
  bool checkCollision_remodeling(unsigned wid, const warp_inst_t *inst) const;
  int find_first_collision_remodeling(unsigned wid,
                                      const warp_inst_t *inst) const;
  bool pendingWrites(unsigned wid) const;
  void printContents() const;
  const bool islongop(unsigned warp_id, unsigned regnum);

  // Qi: register bypass/forwarding network (-is_register_bypass_forwarding_enabled).
  // Models a small number of bypass paths that let a RAW-dependent consumer read a
  // producer's result directly (skipping the register-file writeback round trip) if
  // the consumer issues within `window_cycles` of the producer leaving EX, and a
  // bypass port is still free for that subcore this cycle. WAW hazards (collisions on
  // the issuing instruction's own destination registers) are never bypass-eligible --
  // only true RAW (source-operand) collisions can be forwarded.
  bool checkCollision_remodeling_with_bypass(unsigned wid, const class warp_inst_t *inst,
                                              unsigned int subcore_id,
                                              unsigned long long cur_cycle,
                                              unsigned int window_cycles,
                                              unsigned int ports_per_subcore);
  // valid_from_cycle defaults to "now" (immediately forwardable), matching the original
  // EX-finish call site. Passing a future cycle models a shorter, earlier-stage forward
  // path (e.g. early-forward from a SP_OP's ALU output before its full EX latency
  // elapses) -- see SM::maybe_record_register_bypass_early.
  void recordBypassWrite(unsigned wid, unsigned int reg_id, unsigned long long expire_cycle,
                          unsigned long long valid_from_cycle = 0);

 private:
  void reserveRegister(unsigned wid, unsigned regnum);
  unsigned int get_sid() const { return m_sid; }

  unsigned int m_sid;

  // Qi: bypass/forwarding network state -- see checkCollision_remodeling_with_bypass above.
  // value = {valid_from_cycle, expire_cycle}
  std::vector<std::unordered_map<unsigned int, std::pair<unsigned long long, unsigned long long> > > m_bypass_forward_table;
  std::unordered_map<unsigned int, unsigned int> m_bypass_ports_used_this_cycle;               // [subcore_id]
  std::unordered_map<unsigned int, unsigned long long> m_bypass_ports_last_reset_cycle;        // [subcore_id]

  bool m_is_trace_mode;

  // keeps track of pending writes to registers
  // indexed by warp id, reg_id => pending write count
  std::vector<std::set<unsigned> > reg_table;
  // Register that depend on a long operation (global, local or tex memory)
  std::vector<std::set<unsigned> > longopregs;

  class gpgpu_t *m_gpu;
};

#endif /* SCOREBOARD_H_ */
