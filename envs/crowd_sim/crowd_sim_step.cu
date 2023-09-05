// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause
#include <stdio.h>

__constant__ float kTwoPi = 6.28318530718;
__constant__ float kEpsilon = 1.0e-10;  // to prevent indeterminate cases
__constant__ float kMaxDistance = 1.0e10;

extern "C" {
  // Device helper function to generate observation
  __device__ void CudaCrowdSimGenerateObservation(
      float * obs_arr,
      const int * agent_types_arr,
      float * agent_x_arr,
      const float kAgentXRange,
      float * agent_y_arr,
      const float kAgentYRange,
      float * agent_energy_arr,
      const float kAgentEnergyRange,
      const int kNumTargets,
      const int kNumAgentsObserved,
      const float * target_x_timelist,
      const float * target_y_timelist,
      float * target_aoi_arr,
      float * neighbor_agent_distances_arr,
      int * neighbor_agent_ids_sorted_by_distances_arr,
      const float kDroneCarCommRange,
      int * env_timestep_arr,
      int kNumAgents,
      int kEpisodeLength,
      const int kEnvId,
      const int kThisAgentId,
      const int kThisAgentArrayIdx  
  ) {
    const int num_features = 2 + 2 * kNumAgentsObserved * 2 + 100;
    if (kThisAgentId < kNumAgents) {
      const int kThisAgentIdxOffset = kEnvId * kNumAgents * num_features + kThisAgentId * num_features;
      for (int i=0; i < num_features; i++){ obs_arr[kThisAgentIdxOffset + i] = 0.0;}
      // ------------------------------------
      // [Part 1] self info (2,)
      obs_arr[kThisAgentIdxOffset + 0] = agent_types_arr[kThisAgentId];
      obs_arr[kThisAgentIdxOffset + 1] = (agent_energy_arr[kThisAgentArrayIdx] / kAgentEnergyRange);

      // ------------------------------------
      // [Part 2] other agent's infosw (2 * self.num_agents_observed * 2)
      for (int idx = 0; idx < 2 * kNumAgentsObserved; idx++) {
        obs_arr[kThisAgentIdxOffset + 2 + idx * 2 + 0] = 0.0;
        obs_arr[kThisAgentIdxOffset + 2 + idx * 2 + 1] = 0.0;
      }
      // Sort the neighbor homogeneous and heterogeneous agents as the following part of observations
      const int kThisDistanceArrayIdxOffset = kEnvId * kNumAgents * (kNumAgents-1) + kThisAgentId * (kNumAgents-1);
      int i_index=0;
      for (int agent_idx = 0; agent_idx < kNumAgents; agent_idx++){
        if (agent_idx == kThisAgentId){
          continue;
        }
        float dist = sqrt(
            pow(agent_x_arr[kThisAgentArrayIdx] - agent_x_arr[kEnvId * kNumAgents+agent_idx], 2) +
            pow(agent_y_arr[kThisAgentArrayIdx] - agent_y_arr[kEnvId * kNumAgents+agent_idx], 2)
              );
            
        neighbor_agent_distances_arr[kThisDistanceArrayIdxOffset + i_index] = dist;
        neighbor_agent_ids_sorted_by_distances_arr[kThisDistanceArrayIdxOffset + i_index] = agent_idx;
        i_index++;
      }

      int j_index;  // A simple bubble sort within one gpu thread
      for (int i = 0; i < kNumAgents-2; i++) {  
        for (int j = 0; j < kNumAgents-i-2; j++) {
          j_index = kThisDistanceArrayIdxOffset + j;

          if (neighbor_agent_distances_arr[j_index] > neighbor_agent_distances_arr[j_index+1]) {
            float tmp1 = neighbor_agent_distances_arr[j_index];
            neighbor_agent_distances_arr[j_index] = neighbor_agent_distances_arr[j_index+1];
            neighbor_agent_distances_arr[j_index+1] = tmp1;

            int tmp2 = neighbor_agent_ids_sorted_by_distances_arr[j_index];
            neighbor_agent_ids_sorted_by_distances_arr[j_index] = neighbor_agent_ids_sorted_by_distances_arr[j_index+1];
            neighbor_agent_ids_sorted_by_distances_arr[j_index+1] = tmp2;
          }
        }
      }

      int homoge_part_idx = 0;
      int hetero_part_idx = 0;
      const int kThisHomogeAgentIdxOffset = kEnvId * kNumAgents * num_features + kThisAgentId * num_features + 2;
      const int kThisHeteroAgentIdxOffset = kEnvId * kNumAgents * num_features + kThisAgentId * num_features + 2 + 2 * kNumAgentsObserved;
      for (int i = 0; i < kNumAgents-1; i ++){
        int other_agent_idx = neighbor_agent_ids_sorted_by_distances_arr[kThisDistanceArrayIdxOffset + i];
        // printf("agent %d - other idx: %d\n", kThisAgentId,other_agent_idx);
        if ((agent_types_arr[kThisAgentId] == agent_types_arr[other_agent_idx]) && (homoge_part_idx<kNumAgentsObserved)){
          float delta_x = (agent_x_arr[kEnvId * kNumAgents + other_agent_idx]  - agent_x_arr[kThisAgentArrayIdx])/kAgentXRange;
          float delta_y = (agent_y_arr[kEnvId * kNumAgents + other_agent_idx]  - agent_y_arr[kThisAgentArrayIdx])/kAgentYRange;
          obs_arr[kThisHomogeAgentIdxOffset + homoge_part_idx*2 + 0] = delta_x;
          obs_arr[kThisHomogeAgentIdxOffset + homoge_part_idx*2 + 1] = delta_y;
          homoge_part_idx++;
        }
        if ((agent_types_arr[kThisAgentId] != agent_types_arr[other_agent_idx]) && (hetero_part_idx<kNumAgentsObserved)){
          float delta_x = (agent_x_arr[kEnvId * kNumAgents + other_agent_idx]  - agent_x_arr[kThisAgentArrayIdx])/kAgentXRange;
          float delta_y = (agent_y_arr[kEnvId * kNumAgents + other_agent_idx]  - agent_y_arr[kThisAgentArrayIdx])/kAgentYRange;
          obs_arr[kThisHeteroAgentIdxOffset + hetero_part_idx*2 + 0] = delta_x;
          obs_arr[kThisHeteroAgentIdxOffset + hetero_part_idx*2 + 1] = delta_y;
          hetero_part_idx++;
        }
      }

      // ------------------------------------
      // [Part 3] aoi grid (10 * 10)
      float grid_center_x = agent_x_arr[kThisAgentArrayIdx];
      float grid_center_y = agent_y_arr[kThisAgentArrayIdx];
      const float grid_width = kDroneCarCommRange * 2 / 10;
      float grid_min_x = grid_center_x - 5 * grid_width;
      float grid_min_y = grid_center_y - 5 * grid_width;
      float grid_max_x = grid_center_x + 5 * grid_width;
      float grid_max_y = grid_center_y + 5 * grid_width;
      int grid_point_count[100] = {0}; 
      float temp_aoi_grid[100] = {0.0f};
      const int kThisTargetPositionTimelistIdxOffset = env_timestep_arr[kEnvId] * kNumTargets; 
      const int kThisTargetAgeArrayIdxOffset = kEnvId * kNumTargets;

      for (int i = 0; i < kNumTargets; ++i) {
        int x = floorf((target_x_timelist[kThisTargetPositionTimelistIdxOffset+i] - grid_min_x) / (grid_max_x - grid_min_x) * 10);
        int y = floorf((target_y_timelist[kThisTargetPositionTimelistIdxOffset+i] - grid_min_y) / (grid_max_y - grid_min_y) * 10);

        if (0 <= x && x < 10 && 0 <= y && y < 10) {
            int idx = x * 10 + y;
            grid_point_count[idx] += 1;
            temp_aoi_grid[idx] += target_aoi_arr[kThisTargetAgeArrayIdxOffset+i];
        }
      }

      const int kThisAgentAoIGridIdxOffset = kEnvId * kNumAgents * num_features + kThisAgentId * num_features + 2 + 2 * kNumAgentsObserved * 2;
      for (int i = 0; i < 100; ++i) {
          if (grid_point_count[i] > 0) {
              obs_arr[kThisAgentAoIGridIdxOffset+i] = temp_aoi_grid[i] / grid_point_count[i] / kEpisodeLength;
          } else {
              obs_arr[kThisAgentAoIGridIdxOffset+i] = 0.0;
          }
      }
    }
  }

  // k: const with timesteps, arr: on current timestep, timelist: multiple timesteps
  __global__ void CudaCrowdSimStep(
    float * obs_arr,
    int * action_indices_arr,
    float * rewards_arr,
    const int * agent_types_arr,
    const float * car_action_space_dx_arr,
    const float * car_action_space_dy_arr,
    const float * drone_action_space_dx_arr,
    const float * drone_action_space_dy_arr,
    float * agent_x_arr,
    const float kAgentXRange,
    float * agent_y_arr,
    const float kAgentYRange,
    float * agent_energy_arr,
    const float kAgentEnergyRange,
    const int kNumTargets,
    const int kNumAgentsObserved,
    const float * target_x_timelist,
    const float * target_y_timelist,
    float * target_aoi_arr,
    int * valid_status_arr,
    int * neighbor_agent_ids_arr,
    const float kCarSensingRange,
    const float kDroneSensingRange,
    const float kDroneCarCommRange,    
    float * neighbor_agent_distances_arr,
    int * neighbor_agent_ids_sorted_by_distances_arr,
    int * done_arr,
    int * env_timestep_arr,
    int kNumAgents,
    int kEpisodeLength    
  ) {
    const int kEnvId = getEnvID(blockIdx.x);
    const int kThisAgentId = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
    const int kThisAgentArrayIdx = kEnvId * kNumAgents + kThisAgentId;
    const int kNumActionDim = 1;  // use Discreate instead of MultiDiscrete
    // -------------------------------
    // Update Timestep
    // Increment time ONCE -- only 1 thread can do this.
    if (kThisAgentId == 0) {
      env_timestep_arr[kEnvId] += 1;
    }
    __sync_env_threads(); // Wait here until timestep has been updated
    assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId] <=
      kEpisodeLength);

    // -------------------------------
    // Load Actions to update agent positions
    if (kThisAgentId < kNumAgents) {
      int kThisAgentActionIdxOffset = kEnvId * kNumAgents * kNumActionDim +
        kThisAgentId * kNumActionDim;
      float dx,dy;
      bool is_drone = agent_types_arr[kThisAgentId];
      if (!is_drone){ // Car Movement
        dx = car_action_space_dx_arr[action_indices_arr[kThisAgentActionIdxOffset]];
        dy = car_action_space_dy_arr[action_indices_arr[kThisAgentActionIdxOffset]];
      }
      else{  // Drone Movement
        dx = drone_action_space_dx_arr[action_indices_arr[kThisAgentActionIdxOffset]];
        dy = drone_action_space_dy_arr[action_indices_arr[kThisAgentActionIdxOffset]];
      }
      
      float consume_energy = 0.0;  //TODO: lack uav/ugv energy consumption
      agent_x_arr[kThisAgentArrayIdx] += dx;
      agent_y_arr[kThisAgentArrayIdx] += dy;
      agent_energy_arr[kThisAgentArrayIdx] -= consume_energy;
   
    }
    __sync_env_threads();  // Make sure all agents have updated their positions
    // -------------------------------
    // Compute valid status
    if (kThisAgentId < kNumAgents){
      valid_status_arr[kThisAgentArrayIdx] = 1;
      float min_dist = kMaxDistance;
      bool is_drone = agent_types_arr[kThisAgentId];

      if (is_drone){  // drone
        int nearest_car_id = -1;
        neighbor_agent_ids_arr[kThisAgentArrayIdx] = -1;
        for (int other_agent_id = 0; other_agent_id < kNumAgents; other_agent_id++) {
          bool is_car = !agent_types_arr[other_agent_id];
          if (is_car) {
            float dist = sqrt(
                pow(agent_x_arr[kEnvId * kNumAgents+kThisAgentId] - agent_x_arr[kEnvId * kNumAgents+other_agent_id], 2) +
                pow(agent_y_arr[kEnvId * kNumAgents+kThisAgentId] - agent_y_arr[kEnvId * kNumAgents+other_agent_id], 2));
            
            if (dist < min_dist) {
              min_dist = dist;
              nearest_car_id = other_agent_id;
            }
          }
        }

        if (min_dist <= kDroneCarCommRange) {
          neighbor_agent_ids_arr[kThisAgentArrayIdx] = nearest_car_id;
        }
        else {
          valid_status_arr[kThisAgentArrayIdx] = 0;
        }
        // printf("%d valid: %d, %d\n", kThisAgentId, valid_status_arr[kThisAgentArrayIdx], neighbor_agent_ids_arr[kThisAgentArrayIdx]);
      }
      rewards_arr[kThisAgentArrayIdx] = 0.0;
      
    }
    __sync_env_threads(); // Make sure all agents have updated their valid status
    // printf("%d\n", neighbor_agent_ids_arr[kEnvId * kNumAgents + 5]);

    // -------------------------------
    // Compute reward
    if (kThisAgentId == 0){
      const int kThisTargetAgeArrayIdxOffset = kEnvId * kNumTargets;
      const int kThisTargetPositionTimelistIdxOffset = env_timestep_arr[kEnvId] * kNumTargets;

      for (int target_idx=0; target_idx<kNumTargets; target_idx++){
        float min_dist = kMaxDistance; 
        int nearest_agent_id = -1;
        for (int agent_idx=0; agent_idx < kNumAgents; agent_idx++){
          bool is_valid = valid_status_arr[kEnvId * kNumAgents+agent_idx];
          if (!is_valid){
            continue;
          }
          else{
            float dist = sqrt(
                pow(agent_x_arr[kEnvId * kNumAgents+agent_idx] - target_x_timelist[kThisTargetPositionTimelistIdxOffset+target_idx], 2) +
                pow(agent_y_arr[kEnvId * kNumAgents+agent_idx] - target_y_timelist[kThisTargetPositionTimelistIdxOffset+target_idx], 2)
              );
            
            if (dist < min_dist) {
                min_dist = dist;
                nearest_agent_id = agent_idx;
            }
          }
          // printf("t:%d a:%d valid: %d\n", target_idx, agent_idx, valid_status_arr[kEnvId * kNumAgents+agent_idx]);
        }
        if (min_dist <= kDroneSensingRange){
          bool is_drone = agent_types_arr[nearest_agent_id];
          rewards_arr[kEnvId * kNumAgents + nearest_agent_id] += (target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx]-1) / kEpisodeLength;
          if (is_drone){
            int drone_nearest_car_id = neighbor_agent_ids_arr[kEnvId * kNumAgents + nearest_agent_id];
            rewards_arr[kEnvId * kNumAgents + drone_nearest_car_id] += (target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx]-1) / kEpisodeLength;
            // printf("t:%d a:%d na: %d rew: %f\n", target_idx, nearest_agent_id, drone_nearest_car_id, rewards_arr[kEnvId * kNumAgents + nearest_agent_id]);
          }
          target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx] = 1.0;
        }
        else{
          target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx] += 1.0;
        }
      }
    }
    __sync_env_threads(); // Make sure all agents have calculated the reward

    // -------------------------------
    // Compute Observation
    CudaCrowdSimGenerateObservation(
      obs_arr,
      agent_types_arr,
      agent_x_arr,
      kAgentXRange,
      agent_y_arr,
      kAgentYRange,
      agent_energy_arr,
      kAgentEnergyRange,
      kNumTargets,
      kNumAgentsObserved,
      target_x_timelist,
      target_y_timelist,
      target_aoi_arr,
      neighbor_agent_distances_arr,
      neighbor_agent_ids_sorted_by_distances_arr,
      kDroneCarCommRange,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx);

    __sync_env_threads();  // Wait here to update observation before determining done_arr

    // -------------------------------
    // Use only agent 0's thread to set done_arr
    if (kThisAgentId == 0) {
      if (env_timestep_arr[kEnvId] == kEpisodeLength) {
          done_arr[kEnvId] = 1;
      }
    }
  }
}
