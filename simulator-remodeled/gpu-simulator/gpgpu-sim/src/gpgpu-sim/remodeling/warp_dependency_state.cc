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


#include <cassert>

#include "warp_dependency_state.h"
#include "../shader.h"


Wait_Barrier::Wait_Barrier(unsigned int barrier_id){
    m_barrier_id = barrier_id;
    m_counter = 0;
}

unsigned int Wait_Barrier::get_counter() const{
    return m_counter;
}

unsigned int Wait_Barrier::get_barrier_id(){
    return m_barrier_id;
}

bool Wait_Barrier::is_ready(unsigned int min_val){
    return m_counter <= min_val;
}

void Wait_Barrier::reset(){
    m_counter = 0;
}

void Wait_Barrier::decrease_counter(){
    assert(m_counter > 0);
    m_counter--;
}

void Wait_Barrier::increase_counter(){
    assert(m_counter < 63);
    m_counter++;
}

void Wait_Barrier::print_state(FILE *out){
    fprintf(out, "Barrier %d: %d\n", m_barrier_id, m_counter);
}

Dependency_State::Dependency_State(const shader_core_config* config, shader_core_stats *stats) {
    m_yield = 0;
    m_stall_counter = 0;
    m_num_pending_ldgsts = 0;
    for(unsigned int i = 0; i < config->num_wait_barriers_per_warp; i++){
        m_wait_barriers.push_back(Wait_Barrier(i));
        m_pending_mem_pcs.emplace_back();
    }
    m_stats = stats;
}

void Dependency_State::reset() {
    m_yield = 0;
    m_stall_counter = 0;
    m_num_pending_ldgsts = 0;
    for(auto &wait_barrier : m_wait_barriers){
        wait_barrier.reset();
    }
    for(auto &pending_mem_pc : m_pending_mem_pcs) {
        pending_mem_pc.clear();
    }
    m_pending_ldgsts_pcs.clear();
}

void Dependency_State::cycle() {
    m_yield >>=1;
    m_stall_counter >>=1;
}

void Dependency_State::set_yield() {
    m_yield = 2;
}

void Dependency_State::set_stall_counter(unsigned int stall_counter) {
    m_stall_counter = stall_counter;
}

void Dependency_State::action_over_wait_barrier(Wait_Barrier_Entry_Modifier *wait_barrier_entry_modifier) {
    unsigned int barrier_id = wait_barrier_entry_modifier->barrier_id;
    if(wait_barrier_entry_modifier->barrier_action == Wait_Barrier_Action::INCREASE_COUNTER){
        m_wait_barriers[barrier_id].increase_counter();
        if (wait_barrier_entry_modifier->pc != 0) {
            m_pending_mem_pcs[barrier_id].push_back(wait_barrier_entry_modifier->pc);
        }
    }else if(wait_barrier_entry_modifier->barrier_action == Wait_Barrier_Action::DECREASE_COUNTER){
        m_wait_barriers[barrier_id].decrease_counter();
        if (!m_pending_mem_pcs[barrier_id].empty()) {
            m_pending_mem_pcs[barrier_id].pop_front();
        }
    }else{
        std::cout << "Error: Wait barrier action not recognized" << std::endl;
        abort();
    }
}

bool Dependency_State::is_yield_ready() {
    return m_yield == false;
}

bool Dependency_State::is_stall_counter_0() {
    return m_stall_counter == 0;
}

bool Dependency_State::are_wait_barriers_ready(std::vector<Wait_Barrier_Checking> wait_barriers_checking) {
    for(auto &wait_barrier_checking : wait_barriers_checking){
        if(!m_wait_barriers[wait_barrier_checking.barrier_id].is_ready(wait_barrier_checking.min_val)){
            return false;
        }
    }
    return true;
}

void Dependency_State::increase_num_pending_ldgsts(new_addr_type pc) {
    m_num_pending_ldgsts++;
    if (pc != 0) {
        m_pending_ldgsts_pcs.push_back(pc);
    }
}
void Dependency_State::decrease_num_pending_ldgsts(new_addr_type pc) {
    (void)pc;
    assert(m_num_pending_ldgsts > 0);
    m_num_pending_ldgsts--;
    if (!m_pending_ldgsts_pcs.empty()) {
        m_pending_ldgsts_pcs.pop_front();
    }
}
bool Dependency_State::are_ldgsts_pending() {
    return m_num_pending_ldgsts != 0;
}

unsigned int Dependency_State::get_wait_barrier_counter(unsigned barrier_id) const {
    return m_wait_barriers[barrier_id].get_counter();
}

const std::deque<new_addr_type> &Dependency_State::get_pending_mem_pcs(unsigned barrier_id) const {
    return m_pending_mem_pcs[barrier_id];
}

const std::deque<new_addr_type> &Dependency_State::get_pending_ldgsts_pcs() const {
    return m_pending_ldgsts_pcs;
}

bool Dependency_State::are_pending_dependencies() {
    bool are_wait_barriers_pending = false;
    for(std::size_t i = 0; (i < m_wait_barriers.size()) && !are_wait_barriers_pending; i++){
        if(m_wait_barriers[i].get_counter() > 0){
            are_wait_barriers_pending = true;
        }
    }
    return are_ldgsts_pending() || are_wait_barriers_pending;
}

void Dependency_State::print_state(FILE *out){
    fprintf(out, "Yield: %d\n", m_yield);
    fprintf(out, "Stall counter: %d\n", m_stall_counter);
    for(auto &wait_barrier : m_wait_barriers){
        wait_barrier.print_state(out);
    }
}