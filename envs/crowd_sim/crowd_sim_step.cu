// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause
#include <stdio.h>
#include <math.h>
__constant__ float kTwoPi = 6.28318530718;
__constant__ float kEpsilon = 1.0e-10;  // to prevent indeterminate cases
__constant__ float kMaxDistance = 1.0e10;

extern "C" {
// typedef pair<int, float> dis_pair;
  __device__ float calculateEnergy(const int& slot_time, const float& move_time, const int& agent_speed){
     float stop_time = slot_time - move_time;
     if (agent_speed < 10){
        float idle_cost = 17.49;
        float energy_factor = 7.4;
        return (idle_cost + energy_factor) * agent_speed * move_time + idle_cost * stop_time;
     }
     else{
        float P0 = 79.8563;  // blade profile power, W
        float P1 = 88.6279;  // derived power, W
        float U_tips = 120;  // tip speed of the rotor blade of the UAV,m/s
        float v0 = 4.03;  // the mean rotor induced velocity in the hovering state,m/s
        float d0 = 0.6;  // fuselage drag ratio
        float rho = 1.225;  // density of air,kg/m^3
        float s0 = 0.05;  // the rotor solidity
        float A = 0.503;  // the area of the rotor disk, m^2
        int vt = agent_speed;
        int vt_2 = vt * vt;
        int vt_4 = vt_2 * vt_2;
        float v0_2 = v0 * v0;
        float v0_4 = v0_2 * v0_2;
        float flying_energy = P0 * (1 + 3 * vt_2 / (U_tips * U_tips)) + \
                           P1 * sqrt(sqrt(1 + vt_4 / (4 * v0_4)) - vt_2 / (2 * v0_2)) + \
                           0.5 * d0 * rho * s0 * A * vt_2 * vt;
        return move_time * flying_energy + stop_time * (P0 + P1);
     }
  }
// __device__ int SortCompare(const void* a, const void* b) {
//     const dis_pair* pa = (const dis_pair*) a;
//     const dis_pair* pb = (const dis_pair*) b;
//     if (pa->second < pb->second) return -1;
//     else if (pa->second > pb->second) return 1;
//     else return 0;
// }
__device__ void CUDACrowdSimGenerateAoIGrid(
  float * obs_arr,
  const float grid_center_x,
  const float grid_center_y,
  const int sense_range_x,
  const int sense_range_y,
  const float * target_x_time_list,
  const float * target_y_time_list,
  int * target_aoi_arr,
  const int timestep,
  const int kEnvId,
  const int kThisAgentId,
  const int kThisEnvAgentsOffset,
  const int kNumAgents,
  const int kNumAgentsObserved,
  const int kNumTargets,
  const int kEpisodeLength,
  const int num_features,
  const float max_distance_x,
  const float max_distance_y,
  const float kAgentXRange,
  const float kAgentYRange
) {
      // ------------------------------------
      // [Part 3] aoi grid (10 * 10)
      const float x_width = sense_range_x >> 1;
      const float y_width = sense_range_y >> 1;
      float grid_min_x = grid_center_x - x_width;
      float grid_min_y = grid_center_y - y_width;
      float grid_max_x = grid_center_x + x_width;
      float grid_max_y = grid_center_y + y_width;
      int grid_point_count[100] = {0};
      int temp_aoi_grid[100] = {0};
      const int kThisTargetPositionTimeListIdxOffset = timestep * kNumTargets;
      const int kThisTargetAgeArrayIdxOffset = kEnvId * kNumTargets;

      for (int i = 0; i < kNumTargets; ++i) {
        int x = floorf((target_x_time_list[kThisTargetPositionTimeListIdxOffset+i] - grid_min_x) / (grid_max_x - grid_min_x) * 10);
        int y = floorf((target_y_time_list[kThisTargetPositionTimeListIdxOffset+i] - grid_min_y) / (grid_max_y - grid_min_y) * 10);

        if (0 <= x && x < 10 && 0 <= y && y < 10) {
            int idx = x * 10 + y;
            grid_point_count[idx]++;
            temp_aoi_grid[idx] += target_aoi_arr[kThisTargetAgeArrayIdxOffset+i];
        }
      }
      int kThisAgentAoIGridIdxOffset;
      if (kThisAgentId == -1) {
        kThisAgentAoIGridIdxOffset = kNumAgents << 2;
      }
      else{
        kThisAgentAoIGridIdxOffset = (kThisAgentId + kThisEnvAgentsOffset) * num_features + 2 + (kNumAgentsObserved << 2);
      }
      for (int i = 0; i < 100; ++i) {
        float aoi_value = grid_point_count[i] > 0 ? (temp_aoi_grid[i] * 1.0) / grid_point_count[i] / kEpisodeLength : 0.0;
        obs_arr[kThisAgentAoIGridIdxOffset + i] = aoi_value;
    }
}
  // Device helper function to generate observation
  __device__ void CudaCrowdSimGenerateObservation(
      float * state_arr,
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
      const float * target_x_time_list,
      const float * target_y_time_list,
      int * target_aoi_arr,
//       dis_pair * neighbor_pairs,
      float * neighbor_agent_distances_arr,
      int * neighbor_agent_ids_sorted_by_distances_arr,
      const float kDroneCarCommRange,
      int * env_timestep_arr,
      int kNumAgents,
      int kEpisodeLength,
      const int num_features,
      const int kEnvId,
      const int kThisAgentId,
      const int kThisAgentArrayIdx,
      const int kThisEnvAgentsOffset,
      const float max_distance_x,
      const float max_distance_y
  ) {
    // observation: agent type, agent energy, Heterogeneous and homogeneous visible agents
    // displacements, 100 dim AoI Maps.
    // state: all agents type, energy, position (4dim per agent) + 100 dim AoI Maps.
    const int state_features = (kNumAgents << 2) + 100;
    const int shifted_id = kThisAgentId << 2;
    const int kThisEnvStateOffset = kEnvId * state_features;
    if (kThisAgentId < kNumAgents) {
      const int kThisAgentIdxOffset = kThisEnvAgentsOffset * num_features + kThisAgentId * num_features;
      for (int i = 0; i < num_features; i++){
      obs_arr[kThisAgentIdxOffset + i] = 0.0;
      }
      // ------------------------------------
      // [Part 1] self info (2,)
      const int my_type = agent_types_arr[kThisAgentId];
      const float my_energy = agent_energy_arr[kThisAgentArrayIdx] / kAgentEnergyRange;
      obs_arr[kThisAgentIdxOffset + 0] = my_type;
      obs_arr[kThisAgentIdxOffset + 1] = my_energy;
      // Fill self info into state
      state_arr[kThisEnvStateOffset + shifted_id + 0] = my_type;
      state_arr[kThisEnvStateOffset + shifted_id + 1] = my_energy;
      // ------------------------------------
      // [Part 2] other agent's infos (2 * self.num_agents_observed * 2)
      // Other agents displacements are sorted by distance
      for (int idx = 0; idx < 2 * kNumAgentsObserved; idx++) {
        obs_arr[kThisAgentIdxOffset + 2 + idx * 2 + 0] = 0.0;
        obs_arr[kThisAgentIdxOffset + 2 + idx * 2 + 1] = 0.0;
      }
      // Sort the neighbor homogeneous and heterogeneous agents as the following part of observations

      const int kThisDistanceArrayIdxOffset = (kThisAgentId + kThisEnvAgentsOffset) * (kNumAgents - 1);
      for (int agent_idx = 0; agent_idx < kNumAgents; agent_idx++){
//         dis_pair & current = neighbor_pairs[kThisDistanceArrayIdxOffset + i_index];
        if (agent_idx != kThisAgentId){
        float temp_x = agent_x_arr[kThisAgentArrayIdx] - agent_x_arr[kThisEnvAgentsOffset + agent_idx];
        float temp_y = agent_y_arr[kThisAgentArrayIdx] - agent_y_arr[kThisEnvAgentsOffset + agent_idx];
//         current.first = sqrt(temp_x * temp_x + temp_y * temp_y);
//         current.second = agent_idx;
        neighbor_agent_distances_arr[kThisDistanceArrayIdxOffset + agent_idx] = sqrt(temp_x * temp_x + temp_y * temp_y);
        neighbor_agent_ids_sorted_by_distances_arr[kThisDistanceArrayIdxOffset + agent_idx] = agent_idx;
        }
        //  state stores position of each agents
        state_arr[kThisEnvStateOffset + shifted_id + 2] = agent_x_arr[kThisEnvAgentsOffset + agent_idx] / kAgentXRange;
        state_arr[kThisEnvStateOffset + shifted_id + 3] = agent_y_arr[kThisEnvAgentsOffset + agent_idx] / kAgentYRange;
      }
      int j_index;  // A simple bubble sort within one gpu thread
      for (int i = 0; i < kNumAgentsObserved - 1; i++) {
        for (int j = 0; j < kNumAgentsObserved - i - 1; j++) {
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
    const int kThisHomogeAgentIdxOffset = kThisEnvAgentsOffset * num_features + kThisAgentId * num_features + 2;
    const int kThisHeteroAgentIdxOffset = kThisEnvAgentsOffset * num_features + kThisAgentId * num_features + 2 + 2 * kNumAgentsObserved;

    const float agent_x = agent_x_arr[kThisAgentArrayIdx];
    const float agent_y = agent_y_arr[kThisAgentArrayIdx];
    const int kThisAgentType = agent_types_arr[kThisAgentId];

    for (int i = 0; i < kNumAgentsObserved; i++) {
        int other_agent_idx = neighbor_agent_ids_sorted_by_distances_arr[kThisDistanceArrayIdxOffset + i];
        int other_agent_type = agent_types_arr[other_agent_idx];

        // Precompute delta values to reduce redundancy.
        float delta_x = (agent_x_arr[kThisEnvAgentsOffset + other_agent_idx] - agent_x) / kAgentXRange;
        float delta_y = (agent_y_arr[kThisEnvAgentsOffset + other_agent_idx] - agent_y) / kAgentYRange;

        if (kThisAgentType == other_agent_type && homoge_part_idx < kNumAgentsObserved) {
            obs_arr[kThisHomogeAgentIdxOffset + homoge_part_idx*2 + 0] = delta_x;
            obs_arr[kThisHomogeAgentIdxOffset + homoge_part_idx*2 + 1] = delta_y;
            homoge_part_idx++;
        }

        if (kThisAgentType != other_agent_type && hetero_part_idx < kNumAgentsObserved) {
            obs_arr[kThisHeteroAgentIdxOffset + hetero_part_idx*2 + 0] = delta_x;
            obs_arr[kThisHeteroAgentIdxOffset + hetero_part_idx*2 + 1] = delta_y;
            hetero_part_idx++;
        }
    }

      CUDACrowdSimGenerateAoIGrid(
        obs_arr,
        agent_x_arr[kThisAgentArrayIdx],
        agent_y_arr[kThisAgentArrayIdx],
        kDroneCarCommRange * 2,
        kDroneCarCommRange * 2,
        target_x_time_list,
        target_y_time_list,
        target_aoi_arr,
        env_timestep_arr[kEnvId],
        kEnvId,
        kThisAgentId,
        kThisEnvAgentsOffset,
        kNumAgents,
        kNumAgentsObserved,
        kNumTargets,
        kEpisodeLength,
        num_features,
        max_distance_x,
        max_distance_y,
        kAgentXRange,
        kAgentYRange
      );
  }
}

  // k: const with timesteps, arr: on current timestep, time_list: multiple timesteps
  __global__ void CudaCrowdSimStep(
    float * state_arr,
    float * obs_arr,
    int * action_indices_arr,
    float * rewards_arr,
    float * global_rewards_arr,
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
    const float * target_x_time_list,
    const float * target_y_time_list,
    int * target_aoi_arr,
    bool * target_coverage_arr,
    bool * valid_status_arr,
    int * neighbor_agent_ids_arr,
    const float kCarSensingRange,
    const float kDroneSensingRange,
    const float kDroneCarCommRange,
//     dis_pair * neighbor_pairs,
    float * neighbor_agent_distances_arr,
    int * neighbor_agent_ids_sorted_by_distances_arr,
    int * done_arr,
    int * env_timestep_arr,
    int kNumAgents,
    int kEpisodeLength,
    const int max_distance_x,
    const int max_distance_y,
    const float slot_time,
    const int* agent_speed_arr,
    int dynamic_zero_shot,
    int zero_shot_start
  ) {
    const int kEnvId = getEnvID(blockIdx.x);
    const int kThisAgentId = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
    const int kThisEnvAgentsOffset = kEnvId * kNumAgents;
    const int kThisAgentArrayIdx = kThisEnvAgentsOffset + kThisAgentId;
    const int kNumActionDim = 1;  // use Discrete instead of MultiDiscrete
    // -------------------------------
    // Update Timestep
    // Increment time ONCE -- only 1 thread can do this.
    if (kThisAgentId == 0) {
      int original = env_timestep_arr[kEnvId]++;
      if (original > kEpisodeLength) {
        env_timestep_arr[kEnvId] = 0;
      }
    }
    __sync_env_threads(); // Wait here until timestep has been updated
    assert(env_timestep_arr[kEnvId] > 0 && env_timestep_arr[kEnvId] <=
      kEpisodeLength);
    bool over_range = false;
    // -------------------------------
    // Load Actions to update agent positions
    if (kThisAgentId < kNumAgents) {
      int kThisAgentActionIdxOffset = (kThisEnvAgentsOffset + kThisAgentId) * kNumActionDim;
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

      float new_x = agent_x_arr[kThisAgentArrayIdx] + dx;
      float new_y = agent_y_arr[kThisAgentArrayIdx] + dy;
      if (new_x < max_distance_x && new_y < max_distance_y && new_x > 0 && new_y > 0){
        float distance = sqrt(dx * dx + dy * dy);
        agent_x_arr[kThisAgentArrayIdx] = new_x;
        agent_y_arr[kThisAgentArrayIdx] = new_y;
        int my_speed = agent_speed_arr[is_drone];
        float move_time = distance / my_speed;
        float consume_energy = calculateEnergy(slot_time, move_time, my_speed);
        if (agent_energy_arr[kThisAgentArrayIdx] < consume_energy){
          over_range = true;
          // printf("agent %d out of energy\n", kThisAgentId);
        }
        else{
          agent_energy_arr[kThisAgentArrayIdx] -= consume_energy;
        }
      }
      else{
        over_range = true;
        // printf("agent %d out of bound\n", kThisAgentId);
      }
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
            float temp_x = agent_x_arr[kThisEnvAgentsOffset + kThisAgentId] - \
            agent_x_arr[kThisEnvAgentsOffset + other_agent_id];
            float temp_y = agent_y_arr[kThisEnvAgentsOffset + kThisAgentId] - \
            agent_y_arr[kThisEnvAgentsOffset + other_agent_id];
            float dist = sqrt(temp_x * temp_x + temp_y * temp_y);
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
//         printf("%d valid: %d, %d\n", kThisAgentId, valid_status_arr[kThisAgentArrayIdx], neighbor_agent_ids_arr[kThisAgentArrayIdx]);
      }
      rewards_arr[kThisAgentArrayIdx] = 0.0;
    }
    __sync_env_threads(); // Make sure all agents have updated their valid status
    // printf("%d\n", neighbor_agent_ids_arr[kThisEnvAgentsOffset + 5]);
    // -------------------------------
    // Compute reward
    int count = 0;
    if (kThisAgentId == 0){
      const int kThisTargetAgeArrayIdxOffset = kEnvId * kNumTargets;
      const int kThisTargetPositionTimeListIdxOffset = env_timestep_arr[kEnvId] * kNumTargets;
    for (int target_idx = 0; target_idx < kNumTargets; target_idx++) {
        target_coverage_arr[kThisTargetAgeArrayIdxOffset + target_idx] = false;
        float min_dist = kMaxDistance;
        int nearest_agent_id = -1;
        float target_x = target_x_time_list[kThisTargetPositionTimeListIdxOffset + target_idx];
        float target_y = target_y_time_list[kThisTargetPositionTimeListIdxOffset + target_idx];

        for (int agent_idx = 0; agent_idx < kNumAgents; agent_idx++) {
            bool is_valid = valid_status_arr[kThisEnvAgentsOffset + agent_idx];
            if (is_valid) {
                float temp_x = agent_x_arr[kThisEnvAgentsOffset + agent_idx] - target_x;
                float temp_y = agent_y_arr[kThisEnvAgentsOffset + agent_idx] - target_y;
                float dist = sqrt(temp_x * temp_x + temp_y * temp_y);
//                 printf("%f\n", dist);
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest_agent_id = agent_idx;
                }
            }
        }

        int target_aoi = target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx];
        int reward_increment = (target_aoi - 1);
        if (dynamic_zero_shot && target_idx >= zero_shot_start){
          reward_increment *= 1.5;
        }
        if (min_dist <= kDroneSensingRange && nearest_agent_id != -1) {
            bool is_drone = agent_types_arr[nearest_agent_id];
            rewards_arr[kThisEnvAgentsOffset + nearest_agent_id] += reward_increment;
            if (is_drone) {
                int drone_nearest_car_id = neighbor_agent_ids_arr[kThisEnvAgentsOffset + nearest_agent_id];
                rewards_arr[kThisEnvAgentsOffset + drone_nearest_car_id] += reward_increment;
            }
            target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx] = 1;
            target_coverage_arr[kThisTargetAgeArrayIdxOffset + target_idx] = true;
            count++;
            global_rewards_arr[kEnvId] += reward_increment;
//             printf("target %d covered, coverage arr %d\n", target_idx, target_coverage_arr[kThisTargetAgeArrayIdxOffset + target_idx]);
        } else {
            target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx]++;
            global_rewards_arr[kEnvId]--;
//             printf("target %d not covered, coverage arr %d\n", target_idx, target_coverage_arr[kThisTargetAgeArrayIdxOffset + target_idx]);
        }
    }
    // Normalize rewards
    for(int i = 0;i < kNumAgents;i++){
      rewards_arr[kThisEnvAgentsOffset + i] /= kEpisodeLength;
//       printf("agent %d reward: %f\n", i, rewards_arr[kThisEnvAgentsOffset + i]);
    }
    global_rewards_arr[kEnvId] /= kEpisodeLength;
  }
    __sync_env_threads(); // Make sure all agents have calculated the reward
    const int num_features = 2 + (kNumAgentsObserved << 2) + 100;
    // -------------------------------
    // Compute Observation
    CudaCrowdSimGenerateObservation(
      state_arr,
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
      target_x_time_list,
      target_y_time_list,
      target_aoi_arr,
//       neighbor_pairs,
      neighbor_agent_distances_arr,
      neighbor_agent_ids_sorted_by_distances_arr,
      kDroneCarCommRange,
      env_timestep_arr,
      kNumAgents,
      kEpisodeLength,
      num_features,
      kEnvId,
      kThisAgentId,
      kThisAgentArrayIdx,
      kThisEnvAgentsOffset,
      max_distance_x,
      max_distance_y
      );

    __sync_env_threads();  // Wait here to update observation before determining done_arr
        // const int global_range = kDroneCarCommRange * 4;
        CUDACrowdSimGenerateAoIGrid(
        state_arr,
        max_distance_x >> 1,
        max_distance_y >> 1,
        max_distance_x,
        max_distance_y,
        target_x_time_list,
        target_y_time_list,
        target_aoi_arr,
        env_timestep_arr[kEnvId],
        kEnvId,
        -1,
        kThisEnvAgentsOffset,
        kNumAgents,
        kNumAgentsObserved,
        kNumTargets,
        kEpisodeLength,
        num_features,
        max_distance_x,
        max_distance_y,
        kAgentXRange,
        kAgentYRange
      );
    __sync_env_threads();
    // -------------------------------
    // Use only agent 0's thread to set done_arr
    if (kThisAgentId == 0) {
      if (env_timestep_arr[kEnvId] == kEpisodeLength || over_range) {
          done_arr[kEnvId] = 1;
//           printf("coverage: %d\n", count);
      }
    }
  }
}
