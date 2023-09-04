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
      const int kNumTargetObserved,
      const float * target_x_timelist,
      const float * target_y_timelist,
      const float * target_theta_timelist,
      float * target_aoi_arr,
      float * neighbor_target_distances_arr,
      int * neighbor_target_ids_sorted_by_distance_arr,
      int * env_timestep_arr,
      int kNumAgents,
      int kEpisodeLength,
      const int kEnvId,
      const int kThisAgentId,
      const int kThisAgentArrayIdx  
  ) {
    int num_features = 4;

    if (kThisAgentId < kNumAgents) {
      // Initialize obs to all zeros
      const int kThisAgentIdxOffset = kEnvId * kNumAgents * (kNumAgents + kNumTargetObserved) * num_features +
        kThisAgentId * (kNumAgents + kNumTargetObserved) * num_features;

      for (int idx = 0; idx < kNumAgents + kNumTargetObserved; idx++) {
        obs_arr[kThisAgentIdxOffset + idx * num_features + 0] = 0.0;
        obs_arr[kThisAgentIdxOffset + idx * num_features + 1] = 0.0;
        obs_arr[kThisAgentIdxOffset + idx * num_features + 2] = 0.0;
        obs_arr[kThisAgentIdxOffset + idx * num_features + 3] = 0.0;
      }
      
      // Sort the neighbor targets as the following part of observations
      const int kThisTargetAgeArrayIdxOffset = kEnvId * kNumTargets;
      const int kThisTargetPositionTimelistIdxOffset = env_timestep_arr[kEnvId] * kNumTargets;
      const int kThisTargetDistanceArrayIdxOffset = kEnvId * kNumAgents * kNumTargets + kThisAgentId * kNumTargets;
    
      for (int target_idx = 0; target_idx < kNumTargets; target_idx++){
        float dist = sqrt(
            pow(agent_x_arr[kThisAgentArrayIdx] - target_x_timelist[kThisTargetPositionTimelistIdxOffset+target_idx], 2) +
            pow(agent_y_arr[kThisAgentArrayIdx] - target_y_timelist[kThisTargetPositionTimelistIdxOffset+target_idx], 2)
              );
            
        neighbor_target_ids_sorted_by_distance_arr[kThisTargetDistanceArrayIdxOffset + target_idx] = target_idx;
        neighbor_target_distances_arr[kThisTargetDistanceArrayIdxOffset + target_idx] = dist;
      }

      int j_index;
      for (int i = 0; i < kNumTargets-1; i++) {  // A simple bubble sort within one gpu thread
        for (int j = 0; j < kNumTargets-i-1; j++) {
          j_index = kThisTargetDistanceArrayIdxOffset + j;

          if (neighbor_target_distances_arr[j_index] > neighbor_target_distances_arr[j_index+1]) {
            float tmp1 = neighbor_target_distances_arr[j_index];
            neighbor_target_distances_arr[j_index] = neighbor_target_distances_arr[j_index+1];
            neighbor_target_distances_arr[j_index+1] = tmp1;

            int tmp2 = neighbor_target_ids_sorted_by_distance_arr[j_index];
            neighbor_target_ids_sorted_by_distance_arr[j_index] = neighbor_target_ids_sorted_by_distance_arr[j_index+1];
            neighbor_target_ids_sorted_by_distance_arr[j_index+1] = tmp2;
          }
        }
      }

      for (int idx = 0; idx < kNumAgents + kNumTargetObserved; idx++) {
        if (idx<kNumAgents){
          // obs_arr[kThisAgentIdxOffset + idx * num_features + 0] = agent_x_arr[kEnvId * kNumAgents + idx] / kAgentXRange;
          // obs_arr[kThisAgentIdxOffset + idx * num_features + 1] = agent_y_arr[kEnvId * kNumAgents + idx] / kAgentYRange;
          // obs_arr[kThisAgentIdxOffset + idx * num_features + 2] = static_cast<float>(agent_types_arr[idx]);
          // obs_arr[kThisAgentIdxOffset + idx * num_features + 3] = agent_energy_arr[kEnvId * kNumAgents + idx] / kAgentEnergyRange;
          // for debug
          obs_arr[kThisAgentIdxOffset + idx * num_features + 0] = agent_x_arr[kEnvId * kNumAgents + idx];
          obs_arr[kThisAgentIdxOffset + idx * num_features + 1] = agent_y_arr[kEnvId * kNumAgents + idx];
          obs_arr[kThisAgentIdxOffset + idx * num_features + 2] = agent_types_arr[idx];
          obs_arr[kThisAgentIdxOffset + idx * num_features + 3] = agent_energy_arr[kEnvId * kNumAgents + idx];
        }
        else{
          int target_idx = idx-kNumAgents;
          int neighbor_target_idx = neighbor_target_ids_sorted_by_distance_arr[kThisTargetDistanceArrayIdxOffset + target_idx];
          // obs_arr[kThisAgentIdxOffset + idx * num_features + 0] = target_x_timelist[kThisTargetPositionTimelistIdxOffset + 
          //                                                                                       neighbor_target_idx] / kAgentXRange;
          // obs_arr[kThisAgentIdxOffset + idx * num_features + 1] = target_y_timelist[kThisTargetPositionTimelistIdxOffset + 
          //                                                                                       neighbor_target_idx] / kAgentYRange;
          // obs_arr[kThisAgentIdxOffset + idx * num_features + 2] = target_theta_timelist[kThisTargetPositionTimelistIdxOffset + 
          //                                                                                       neighbor_target_idx] / kTwoPi;
          // obs_arr[kThisAgentIdxOffset + idx * num_features + 3] = target_aoi_arr[kThisTargetAgeArrayIdxOffset + 
          //                                                                                       neighbor_target_idx] / kEpisodeLength;
          // for debug
          obs_arr[kThisAgentIdxOffset + idx * num_features + 0] = target_x_timelist[kThisTargetPositionTimelistIdxOffset + 
                                                                                                neighbor_target_idx];
          obs_arr[kThisAgentIdxOffset + idx * num_features + 1] = target_y_timelist[kThisTargetPositionTimelistIdxOffset + 
                                                                                                neighbor_target_idx];
          obs_arr[kThisAgentIdxOffset + idx * num_features + 2] = target_theta_timelist[kThisTargetPositionTimelistIdxOffset + 
                                                                                                neighbor_target_idx];
          obs_arr[kThisAgentIdxOffset + idx * num_features + 3] = target_aoi_arr[kThisTargetAgeArrayIdxOffset + 
                                                                                                neighbor_target_idx];
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
    const int kNumTargetObserved,
    const float * target_x_timelist,
    const float * target_y_timelist,
    const float * target_theta_timelist,
    float * target_aoi_arr,
    int * valid_status_arr,
    int * neighbor_agent_ids_arr,
    const float kCarSensingRange,
    const float kDroneSensingRange,
    const float kDroneCarCommRange,    
    float * neighbor_target_distances_arr,
    int * neighbor_target_ids_sorted_by_distance_arr,
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
      
      float consume_energy=0;  //TODO: lack uav/ugv energy consumption
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
      }
      rewards_arr[kThisAgentArrayIdx] = 0.0;
      // printf("%d valid: %d, %f, %f, %f\n", kThisAgentId, valid_status_arr[kThisAgentArrayIdx], min_dist, kDroneCarCommRange, kDroneSensingRange);
    }
    __sync_env_threads(); // Make sure all agents have updated their valid status
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
        }
        if (min_dist <= kDroneSensingRange){
          bool is_drone = agent_types_arr[nearest_agent_id];
          rewards_arr[kEnvId * kNumAgents + nearest_agent_id] += (target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx]-1) / kEpisodeLength;
          if (is_drone){
            int drone_nearest_car_id = neighbor_agent_ids_arr[kThisAgentArrayIdx];
            rewards_arr[kEnvId * kNumAgents+drone_nearest_car_id] += (target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx]-1) / kEpisodeLength;
          }
          target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx] = 1;
        }
        else{
          target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx] +=1;
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
      kNumTargetObserved,
      target_x_timelist,
      target_y_timelist,
      target_theta_timelist,
      target_aoi_arr,
      neighbor_target_distances_arr,
      neighbor_target_ids_sorted_by_distance_arr,
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
