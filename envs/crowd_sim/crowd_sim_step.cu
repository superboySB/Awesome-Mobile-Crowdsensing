// Copyright (c) 2021, salesforce.com, inc.
// All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// For full license text, see the LICENSE file in the repo root
// or https://opensource.org/licenses/BSD-3-Clause
#include <stdio.h>

#include <math.h>

__constant__ float kTwoPi = 6.28318530718;
__constant__ float kEpsilon = 1.0e-10; // to prevent indeterminate cases
__constant__ float kMaxDistance = 1.0e10;

extern "C" {
  // typedef pair<int, float> dis_pair;
  __device__ void deviceCopy(float * dest,
    const float * src, int size) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      dest[i] = src[i];
    }
  }
  __device__ float calculateEnergy(const int & slot_time,
    const float & move_time,
      const int & agent_speed) {
    float stop_time = slot_time - move_time;
    if (agent_speed < 10) {
      float idle_cost = 17.49;
      float energy_factor = 7.4;
      return (idle_cost + energy_factor) * agent_speed * move_time + idle_cost * stop_time;
    } else {
      float P0 = 79.8563; // blade profile power, W
      float P1 = 88.6279; // derived power, W
      float U_tips = 120; // tip speed of the rotor blade of the UAV,m/s
      float v0 = 4.03; // the mean rotor induced velocity in the hovering state,m/s
      float d0 = 0.6; // fuselage drag ratio
      float rho = 1.225; // density of air,kg/m^3
      float s0 = 0.05; // the rotor solidity
      float A = 0.503; // the area of the rotor disk, m^2
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
  __device__ void CUDACrowdSimGenerateAoIGrid(
    float * obs_arr,
    const float grid_center_x,
      const float grid_center_y,
        const int sense_range_x,
          const int sense_range_y,
            const float * target_x_time_list,
              const float * target_y_time_list,
                const int * aoi_schedule,
                  int * target_aoi_arr,
                  const int kNumTargets,
                    const int kEpisodeLength,
                      const int dynamic_zero_shot,
                        const int zero_shot_start,
                          const int env_timestep,
                            const int kThisAgentId,
                              const int kEnvId
  ) {
    //     printf("Grid Center: (%f, %f)\n", grid_center_x, grid_center_y);
    const float invEpisodeLength = 1.0f / kEpisodeLength;
    // ------------------------------------
    // [Part 3] aoi grid (10 * 10)
    const float x_width = sense_range_x >> 1;
    const float y_width = sense_range_y >> 1;
    float grid_min_x = grid_center_x - x_width;
    float grid_min_y = grid_center_y - y_width;
    float grid_max_x = grid_center_x + x_width;
    float grid_max_y = grid_center_y + y_width;
    const float inv_delta_x = 10.0 / (grid_max_x - grid_min_x);
    const float inv_delta_y = 10.0 / (grid_max_y - grid_min_y);
    int grid_point_count[100] = {
      0
    };
    int temp_aoi_grid[100] = {
      0
    };
    for (int i = 0; i < kNumTargets; ++i) {
      int is_dyn_point = dynamic_zero_shot && i >= zero_shot_start;
      int x = floorf((target_x_time_list[i] - grid_min_x) * inv_delta_x);
      int y = floorf((target_y_time_list[i] - grid_min_y) * inv_delta_y);
      if (0 <= x && x < 10 && 0 <= y && y < 10) {
//         printf("In Range Target %d: (%f, %f) -> (%d, %d)\n", i, target_x_time_list[i], target_y_time_list[i], x, y);
        int idx = x * 10 + y;
        if (is_dyn_point) {
          if (env_timestep >= aoi_schedule[i - zero_shot_start]) {
            grid_point_count[idx]++;
            temp_aoi_grid[idx] += target_aoi_arr[i] * 5;
          }
        } else {
          grid_point_count[idx]++;
          temp_aoi_grid[idx] += target_aoi_arr[i];
        }
      }
    }
    //       printf("AoI Gen Dest: %p\n", obs_arr);26
    for (int i = 0; i < 100; ++i) {
      obs_arr[i] = grid_point_count[i] > 0 ? (temp_aoi_grid[i] * 1.0) / grid_point_count[i] * invEpisodeLength : 0.0;
      if (obs_arr[i] > 0.0) {
        //         printf("%d %d Total Points in Grid %d: %d, Mean Normalized AoI: %f\n",
        //         kEnvId, kThisAgentId, i, grid_point_count[i], obs_arr[i]);
      }
    }
  }

  __device__ void CUDABubbleSortFloatWithArg(
    float * metricArray,
    int * indexArray,
    const int arraySize
  ) {
    int threadId = threadIdx.x;
    if (threadId == 0) {
      for (int i = 0; i < arraySize; i++) {
        for (int j = 0; j < arraySize - i; j++) {
          if (metricArray[j] > metricArray[j + 1]) {
            float tmp1 = metricArray[j];
            metricArray[j] = metricArray[j + 1];
            metricArray[j + 1] = tmp1;

            int tmp2 = indexArray[j];
            indexArray[j] = indexArray[j + 1];
            indexArray[j + 1] = tmp2;
          }
        }
      }
    }
  }

  __device__ void CUDABubbleSortIntWithArg(
    int * metricArray,
    int * indexArray,
    const int arraySize
  ) {
    int threadId = threadIdx.x;
    if (threadId == 0) {
      for (int i = 0; i < arraySize; i++) {
        for (int j = 0; j < arraySize - i; j++) {
          if (metricArray[j] > metricArray[j + 1]) {
            int tmp1 = metricArray[j];
            metricArray[j] = metricArray[j + 1];
            metricArray[j + 1] = tmp1;

            int tmp2 = indexArray[j];
            indexArray[j] = indexArray[j + 1];
            indexArray[j + 1] = tmp2;
          }
        }
      }
    }
  }

  __device__ void CudaCrowdSimGenerateEmergencyQueue(
    float * emergency_queue,
    int * emergency_index,
    float * emergency_dis,
    const int emergency_count,
      const int EmergencyQueueLength,
        const int FeaturesInEmergencyQueue,
          const float * target_x_time_list,
            const float * target_y_time_list,
              const float agent_x,
                const float agent_y,
                  const int * aoi_schedule,
                    int * target_aoi_arr,
                    bool * target_coverage_arr,
                    const int kNumTargets,
                      const int kEpisodeLength,
                        const int dynamic_zero_shot,
                          const int zero_shot_start,
                            const int env_timestep,
                              const int kThisTargetAgeArrayIdxOffset,
                                const int kThisTargetPositionTimeListIdxOffset,
                                  const int kAgentXRange,
                                    const int kAgentYRange,
                                      const int kThisAgentId,
                                        const int kEnvId
  ) {
    // generate emergency points information
    float invKEpisodeLength = 1.0 / kEpisodeLength;
    memset(emergency_index, -1, sizeof(int) * emergency_count);
    float invKAgentXRange = 1.0 / kAgentXRange;
    float invKAgentYRange = 1.0 / kAgentYRange;
    for (int i = zero_shot_start; i < kNumTargets; i++) {
      //       Condition for putting Emergency into the queue:
      //       1. dynamic_zero_shot mode enabled
      //       2. current timestep is larger than the emergency point's schedule
      //       3. the emergency point is not covered
      // print information of this point
      //       printf("Emergency %d Pos: %f, %f Schedule: %d Coverage: %d\n", i, target_x_time_list[kThisTargetPositionTimeListIdxOffset + i],
      //       target_y_time_list[kThisTargetPositionTimeListIdxOffset + i], aoi_schedule[i - zero_shot_start],
      //       target_coverage_arr[kThisTargetAgeArrayIdxOffset + i]);
      int real_index = i - zero_shot_start;
      if (dynamic_zero_shot && env_timestep > aoi_schedule[real_index] &&
        target_coverage_arr[kThisTargetAgeArrayIdxOffset + i] == false) {
        int pos_index = kThisTargetPositionTimeListIdxOffset + i;
        emergency_index[real_index] = i;
        float delta_x = (target_x_time_list[pos_index] - agent_x) * invKAgentXRange;
        float delta_y = (target_y_time_list[pos_index] - agent_y) * invKAgentYRange;
        emergency_dis[real_index] = sqrt(delta_x * delta_x + delta_y * delta_y);
      } else {
        emergency_dis[real_index] = kMaxDistance;
      }
      //       printf("Emergency Dis Value %p\n", emergency_dis + real_index);
    }
    //       printf("Emergency Queue for Agent %d in Env %d: \n", kThisAgentId, kEnvId);
    CUDABubbleSortFloatWithArg(emergency_dis, emergency_index, emergency_count);
    int total_size = EmergencyQueueLength * FeaturesInEmergencyQueue;
    // Fill the Emergency Queue, but limit to first 10 entries.
    for (int i = 0; i < total_size; i += FeaturesInEmergencyQueue) {
      int real_index = i / FeaturesInEmergencyQueue;
      int pos_index = kThisTargetPositionTimeListIdxOffset + emergency_index[real_index];
      if (real_index < emergency_count && emergency_index[real_index] != -1) {
        emergency_queue[i] = target_x_time_list[pos_index] * invKAgentXRange;
        emergency_queue[i + 1] = target_y_time_list[pos_index] * invKAgentYRange;
        emergency_queue[i + 2] = target_aoi_arr[kThisTargetAgeArrayIdxOffset + emergency_index[real_index]] * invKEpisodeLength;
        emergency_queue[i + 3] = emergency_dis[real_index];
        // print filled information
        //           printf("aoi info: %d %f\n", emergency_index[real_index], emergency_queue[i + 2]);
      } else {
        for (int j = 0; j < FeaturesInEmergencyQueue; j++) {
          emergency_queue[i + j] = 0.0;
        }
      }
    }
  }
// WARNING: This function is not tested.
  __device__ void CudaCrowdSimGreedyAllocation(
  float * agent_x_arr,
  float * agent_y_arr,
  float target_x,
  float target_y,
  int target_idx,
  float * this_emergency_dis_to_target,
  int * this_emergency_dis_to_target_index,
  int * this_emergency_allocation_table,
  int kNumAgents,
  int kThisEnvAgentsOffset
//   int kThisAgentId,
//  int kEnvId,
//   int env_timestep,
  ){
                // calculate distance between current target (x,y) and all agents
              for (int i = 0; i < kNumAgents; i++) {
                float temp_x = target_x - agent_x_arr[kThisEnvAgentsOffset + i];
                float temp_y = target_y - agent_y_arr[kThisEnvAgentsOffset + i];
                float dist = sqrt(temp_x * temp_x + temp_y * temp_y);
                this_emergency_dis_to_target[i] = dist;
                this_emergency_dis_to_target_index[i] = i;
              }
              // sort the distance array as well as the index
              CUDABubbleSortFloatWithArg(this_emergency_dis_to_target, this_emergency_dis_to_target_index, kNumAgents);

//               printf("allocating emergency %d in env %d\n", target_idx, kEnvId);
//               for (int i = 0; i < kNumAgents; i++) {
//                 printf("%d:%d ",kEnvId, this_emergency_allocation_table[i]);
//               }
//               if(kThisAgentId == 0){
//                 printf("\n");
//               }
// using this_emergency_dis_to_target_index, try allocate emergency point to an agent, ignore if all agents are occupied
              for (int i = 0; i < kNumAgents; i++) {
                int candidate_agent_id = this_emergency_dis_to_target_index[i];
                if (this_emergency_allocation_table[candidate_agent_id] == -1) {
                  // allocate this emergency point to this agent
                  this_emergency_allocation_table[candidate_agent_id] = target_idx;
//                   printf("%d: emergency %d at %f,%f in env %d will be handled by %d \n",
//                   env_timestep, target_idx, target_x, target_y, kEnvId, candidate_agent_id);
                  break;
                }
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
                    const int * aoi_schedule,
                      int * target_aoi_arr,
                      int * emergency_index,
                      float * emergency_dis,
                      bool * target_coverage_arr,
                      const int total_num_grids,
                        float * neighbor_agent_distances_arr,
                        int * neighbor_agent_ids_sorted_by_distances_arr,
                        const float kDroneCarCommRange,
                          int env_timestep,
                          int kNumAgents,
                          int kEpisodeLength,
                          const int obs_features,
                            const int obs_vec_features,
                              const int kEnvId,
                                const int kThisAgentId,
                                  const int kThisAgentArrayIdx,
                                    const int AgentFeature,
                                      const int kThisEnvAgentsOffset,
                                        const int kThisEnvStateOffset,
                                          const int state_vec_features,
                                            const float max_distance_x,
                                              const float max_distance_y,
                                                const int dynamic_zero_shot,
                                                  const int zero_shot_start,
                                                    const int emergency_count,
                                                      const int FeaturesInEmergencyQueue,
                                                        const int EmergencyQueueLength
  ) {
    // observation: agent type, agent energy, Heterogeneous and homogeneous visible agents
    // displacements, 100 dim AoI Maps.
    // state: all agents type, energy, position (4dim per agent) + 100 dim AoI Maps.
    //       printf("StateGen: %d %d\n", kThisAgentId, kThisEnvStateOffset);
    const int kThisAgentObsOffset = kThisAgentArrayIdx * obs_features;
    const int kThisAgentAoIGridIdxOffset = kThisAgentObsOffset + obs_vec_features;
    const int kThisAgentFeaturesOffset = AgentFeature * kThisAgentId;
    const int kThisDistanceArrayIdxOffset = kThisAgentArrayIdx * (kNumAgents - 1);
    const float agent_x = agent_x_arr[kThisAgentArrayIdx];
    const float agent_y = agent_y_arr[kThisAgentArrayIdx];
    float * this_state_arr_pointer = state_arr + kThisEnvStateOffset + kThisAgentFeaturesOffset;
    float * this_obs_arr_pointer = obs_arr + kThisAgentObsOffset;
    memset(obs_arr + kThisAgentObsOffset, 0, obs_vec_features * sizeof(float));
    // ------------------------------------
    // [Part 1] self info (4 + kNumAgents, one_hot, type, energy, x, y)
    const int my_type = agent_types_arr[kThisAgentId];
    const float my_energy = agent_energy_arr[kThisAgentArrayIdx] / kAgentEnergyRange;
    // One hot Representation
    //       printf("One Hot for %d\n", kThisAgentId);
    this_obs_arr_pointer[kThisAgentId] = 1;
    // type and energy
    this_obs_arr_pointer[kNumAgents + 0] = my_type;
    this_obs_arr_pointer[kNumAgents + 1] = my_energy;
    // Fill self info into state
    //       printf("State for Agent %d: %d %f %f %f\n", kThisAgentId, my_type, my_energy,
    //       agent_x_arr[kThisAgentArrayIdx] / kAgentXRange, agent_y_arr[kThisAgentArrayIdx] / kAgentYRange);
    this_state_arr_pointer[kThisAgentId] = 1;
    this_state_arr_pointer[kNumAgents + 0] = my_type;
    this_state_arr_pointer[kNumAgents + 1] = my_energy;
    // ------------------------------------
    // [Part 2] other agent's infos (2 * self.num_agents_observed * 2)
    // Other agents displacements are sorted by distance
    // Sort the neighbor homogeneous and heterogeneous agents as the following part of observations

    for (int agent_idx = 0; agent_idx < kNumAgents; agent_idx++) {
      if (agent_idx != kThisAgentId) {
        float temp_x = agent_x - agent_x_arr[kThisEnvAgentsOffset + agent_idx];
        float temp_y = agent_y - agent_y_arr[kThisEnvAgentsOffset + agent_idx];
        neighbor_agent_distances_arr[kThisDistanceArrayIdxOffset + agent_idx] = sqrt(temp_x * temp_x + temp_y * temp_y);
        neighbor_agent_ids_sorted_by_distances_arr[kThisDistanceArrayIdxOffset + agent_idx] = agent_idx;
      } else {
        float normalized_x = agent_x / kAgentXRange;
        float normalized_y = agent_y / kAgentYRange;
        this_state_arr_pointer[kNumAgents + 2] = normalized_x;
        this_state_arr_pointer[kNumAgents + 3] = normalized_y;
        //  state stores position of each agents
        this_obs_arr_pointer[kNumAgents + 2] = normalized_x;
        this_obs_arr_pointer[kNumAgents + 3] = normalized_y;
      }
    }

    CUDABubbleSortFloatWithArg(
      neighbor_agent_distances_arr + kThisDistanceArrayIdxOffset,
      neighbor_agent_ids_sorted_by_distances_arr + kThisDistanceArrayIdxOffset,
      kNumAgentsObserved - 1
    );

    int homoge_part_idx = 0;
    int hetero_part_idx = 0;
    const int kThisHomogeAgentIdxOffset = kThisAgentObsOffset + AgentFeature;
    const int kThisHeteroAgentIdxOffset = kThisHomogeAgentIdxOffset + 2 * kNumAgentsObserved;
    const int kThisAgentType = agent_types_arr[kThisAgentId];
    const int kThisTargetPositionTimeListIdxOffset = env_timestep * kNumTargets;
    const int kThisTargetAgeArrayIdxOffset = kEnvId * kNumTargets;

    for (int i = 0; i < kNumAgentsObserved; i++) {
      int other_agent_idx = neighbor_agent_ids_sorted_by_distances_arr[kThisDistanceArrayIdxOffset + i];
      int other_agent_type = agent_types_arr[other_agent_idx];

      // Precompute delta values to reduce redundancy.
      float delta_x = (agent_x_arr[kThisEnvAgentsOffset + other_agent_idx] - agent_x) / kAgentXRange;
      float delta_y = (agent_y_arr[kThisEnvAgentsOffset + other_agent_idx] - agent_y) / kAgentYRange;

      if (kThisAgentType == other_agent_type && homoge_part_idx < kNumAgentsObserved) {
        obs_arr[kThisHomogeAgentIdxOffset + homoge_part_idx * 2 + 0] = delta_x;
        obs_arr[kThisHomogeAgentIdxOffset + homoge_part_idx * 2 + 1] = delta_y;
        homoge_part_idx++;
      }

      if (kThisAgentType != other_agent_type && hetero_part_idx < kNumAgentsObserved) {
        obs_arr[kThisHeteroAgentIdxOffset + hetero_part_idx * 2 + 0] = delta_x;
        obs_arr[kThisHeteroAgentIdxOffset + hetero_part_idx * 2 + 1] = delta_y;
        hetero_part_idx++;
      }
    }
    // Generate Local AoI Grid of each agent
    CUDACrowdSimGenerateAoIGrid(
      obs_arr + kThisAgentAoIGridIdxOffset,
      agent_x,
      agent_y,
      kDroneCarCommRange * 2,
      kDroneCarCommRange * 2,
      target_x_time_list + kThisTargetPositionTimeListIdxOffset,
      target_y_time_list + kThisTargetPositionTimeListIdxOffset,
      aoi_schedule,
      target_aoi_arr + kThisTargetAgeArrayIdxOffset,
      zero_shot_start,
      kEpisodeLength,
      false,
      zero_shot_start,
      env_timestep,
      kThisAgentId,
      kEnvId
    );
  }

  __device__ int GetNearestAgentId(
    bool * valid_status_arr,
    float target_x,
    float target_y,
    float * agent_x_arr,
    float * agent_y_arr,
    float * original_min_dist,
    int kNumAgents
  ) {
    // add single thread restriction as needed
    float min_dist = kMaxDistance;
    int nearest_agent_id = -1;
    for (int agent_idx = 0; agent_idx < kNumAgents; agent_idx++) {
      bool is_valid = valid_status_arr[agent_idx];
      if (is_valid) {
        float temp_x = agent_x_arr[agent_idx] - target_x;
        float temp_y = agent_y_arr[agent_idx] - target_y;
        float dist = __fsqrt_rn(temp_x * temp_x + temp_y * temp_y); // Using fast sqrt
        if (dist < min_dist) {
          min_dist = dist;
          nearest_agent_id = agent_idx;
        }
      }
    }
    * original_min_dist = min_dist;
    return nearest_agent_id;
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
                        float * target_x_time_list,
                        float * target_y_time_list,
                          const int * aoi_schedule,
                            const int emergency_per_gen,
                              int * emergency_allocation_table,
                              int * target_aoi_arr,
                              int * emergency_index,
                              float * emergency_dis,
                              int * emergency_dis_to_target_index,
                              float * emergency_dis_to_target,
                              bool * target_coverage_arr,
                              bool * valid_status_arr,
                              int * neighbor_agent_ids_arr,
                              const float kCarSensingRange,
                                const float kDroneSensingRange,
                                  const float kDroneCarCommRange,
                                    float * neighbor_agent_distances_arr,
                                    int * neighbor_agent_ids_sorted_by_distances_arr,
                                    int * done_arr,
                                    int * env_timestep_arr,
                                    int kNumAgents,
                                    int kEpisodeLength,
                                    const int max_distance_x,
                                      const int max_distance_y,
                                        const float slot_time,
                                          const int * agent_speed_arr,
                                            int dynamic_zero_shot,
                                            int zero_shot_start,
                                            int single_type_agent,
                                            bool * agents_over_range
  ) {
    //     printf("state: %p, obs: %p\n", state_arr, obs_arr);
    const int kEnvId = getEnvID(blockIdx.x);
    const int kThisAgentId = getAgentID(threadIdx.x, blockIdx.x, blockDim.x);
    const int emergency_count = kNumTargets - zero_shot_start;
    float mean_emergency_aoi = 0.0;
    // Update Timestep
    // Increment time ONCE -- only 1 thread can do this.
    if (kThisAgentId == 0) {
      int original = env_timestep_arr[kEnvId]++;
      if (original > kEpisodeLength) {
        env_timestep_arr[kEnvId] = 0;
      }
    }
    __sync_env_threads(); // Wait here until timestep has been updated
    const int env_timestep = env_timestep_arr[kEnvId];
    //     printf("Agent %d receive timestep: %d\n", kThisAgentId, env_timestep);
    assert(env_timestep > 0 && env_timestep <= kEpisodeLength);
    const int kThisEnvAgentsOffset = kEnvId * kNumAgents;
    const int kThisAgentArrayIdx = kThisEnvAgentsOffset + kThisAgentId;
    const int kNumActionDim = 2; // use Discrete instead of MultiDiscrete
    // Update on 2024.1.2, Double AoI Grid (100 -> 200)
    // Update on 2024.1.10, remove emergency grid. (200 -> 100)
    const int grid_flatten_size = 100;
    const int total_num_grids = grid_flatten_size;
    const int AgentFeature = 4 + kNumAgents;
    // Update on 2024.1.10, add emergency points queue
    const int EmergencyQueueLength = 10;
    const int FeaturesInEmergencyQueue = 4;
    const int StateFullAgentFeature = kNumAgents * AgentFeature;
    const int state_vec_features = StateFullAgentFeature;
    const int state_features = state_vec_features + total_num_grids;
    const int obs_vec_features = AgentFeature + (kNumAgentsObserved << 2) + FeaturesInEmergencyQueue;
    const int obs_features = obs_vec_features + total_num_grids;
    const int kThisEnvStateOffset = kEnvId * state_features;
    int * this_emergency_allocation_table = emergency_allocation_table + kThisEnvAgentsOffset;
//     int * this_emergency_dis_to_target_index = emergency_dis_to_target_index + kThisEnvAgentsOffset;
//     float * this_emergency_dis_to_target = emergency_dis_to_target + kThisEnvAgentsOffset;

    //     printf("Drone Sensing Range: %f\n", kDroneSensingRange);
    //     printf("features: %d, obs: %d\n", state_features, obs_features);
    //     printf("total targets: %d fix targets: %d\n", kNumTargets, zero_shot_start);
    // -------------------------------
    // Load Actions to update agent positions
    if (kThisAgentId < kNumAgents) {
      int kThisAgentActionIdxOffset = kThisAgentArrayIdx * kNumActionDim;
      float dx, dy;
      bool is_drone = agent_types_arr[kThisAgentId];
      if (!is_drone) { // Car Movement
        dx = car_action_space_dx_arr[action_indices_arr[kThisAgentActionIdxOffset]];
        dy = car_action_space_dy_arr[action_indices_arr[kThisAgentActionIdxOffset]];
      } else { // Drone Movement
        dx = drone_action_space_dx_arr[action_indices_arr[kThisAgentActionIdxOffset]];
        dy = drone_action_space_dy_arr[action_indices_arr[kThisAgentActionIdxOffset]];
      }

      float new_x = agent_x_arr[kThisAgentArrayIdx] + dx;
      float new_y = agent_y_arr[kThisAgentArrayIdx] + dy;
      if (new_x < max_distance_x && new_y < max_distance_y && new_x > 0 && new_y > 0) {
        float distance = sqrt(dx * dx + dy * dy);
        agent_x_arr[kThisAgentArrayIdx] = new_x;
        agent_y_arr[kThisAgentArrayIdx] = new_y;
        int my_speed = agent_speed_arr[is_drone];
        float move_time = distance / my_speed;
        float consume_energy = calculateEnergy(slot_time, move_time, my_speed);
        // printf("agent %d out of energy\n", kThisAgentId);
        agent_energy_arr[kThisAgentArrayIdx] -= consume_energy;
      } else {
        agents_over_range[kThisAgentArrayIdx] = true;
        //         printf("%d agent %d out of bound\n", kEnvId, kThisAgentId);
      }
    }
    __sync_env_threads(); // Make sure all agents have updated their positions
    // -------------------------------
    // Compute valid status
    if (kThisAgentId < kNumAgents) {
      valid_status_arr[kThisAgentArrayIdx] = 1;
      bool is_drone = agent_types_arr[kThisAgentId];
      if (is_drone && !single_type_agent) { // drone
        float min_dist = kMaxDistance;
        float my_x = agent_x_arr[kThisAgentArrayIdx];
        float my_y = agent_y_arr[kThisAgentArrayIdx];
        int nearest_car_id = -1;
        neighbor_agent_ids_arr[kThisAgentArrayIdx] = -1;
        for (int other_agent_id = 0; other_agent_id < kNumAgents; other_agent_id++) {
          bool is_car = !agent_types_arr[other_agent_id];
          if (is_car) {
            float temp_x = my_x - agent_x_arr[kThisEnvAgentsOffset + other_agent_id];
            float temp_y = my_y - agent_y_arr[kThisEnvAgentsOffset + other_agent_id];
            float dist = sqrt(temp_x * temp_x + temp_y * temp_y);
            if (dist < min_dist) {
              min_dist = dist;
              nearest_car_id = other_agent_id;
            }
          }
        }
        if (min_dist <= kDroneCarCommRange) {
          neighbor_agent_ids_arr[kThisAgentArrayIdx] = nearest_car_id;
        } else {
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
    //     int count = 0;
    const int kThisTargetAgeArrayIdxOffset = kEnvId * kNumTargets;
    const int kThisTargetPositionTimeListIdxOffset = env_timestep * kNumTargets;
    const float invEpisodeLength = 1.0f / kEpisodeLength;
    if (kThisAgentId == 0) {
      //     printf("TargetTimeListOffset: %d\n", kThisTargetPositionTimeListIdxOffset);
      // print last 30 entries of coverage array
      //     for (int i = 0; i < 30; i++){
      //       printf("%d ", target_coverage_arr[kThisTargetAgeArrayIdxOffset + kNumTargets - 30 + i]);
      //     }
      //     printf("\n");
      float global_reward = 0.0;
      //     int emergency_cover_num = 0;
      //     int valid_emergency_count = 0;
      for (int target_idx = 0; target_idx < kNumTargets; target_idx++) {
        float target_x = target_x_time_list[kThisTargetPositionTimeListIdxOffset + target_idx];
        float target_y = target_y_time_list[kThisTargetPositionTimeListIdxOffset + target_idx];
        int is_dyn_point = dynamic_zero_shot && target_idx >= zero_shot_start;
        bool target_coverage;
        if (!is_dyn_point) {
          target_coverage = false;
        } else {
          if (env_timestep < aoi_schedule[target_idx - zero_shot_start]) {
            // directly skip the target if it is not on schedule yet.
            //           printf("continuing loop for target %d in %d\n", target_idx, kEnvId);
            continue;
          }
          //           printf("Coverage Status for Emergency %d: %d\n", target_idx, target_coverage_arr[kThisTargetAgeArrayIdxOffset + target_idx]);
          target_coverage = target_coverage_arr[kThisTargetAgeArrayIdxOffset + target_idx];
          //           valid_emergency_count++;
        }

        int target_aoi = target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx];
        float min_dist;
        int nearest_agent_id = GetNearestAgentId(
          valid_status_arr + kThisEnvAgentsOffset,
          target_x,
          target_y,
          agent_x_arr + kThisEnvAgentsOffset,
          agent_y_arr + kThisEnvAgentsOffset,
          &min_dist,
          kNumAgents
        );
        int reward_increment = (target_aoi - 1);
        if (is_dyn_point) {
          reward_increment *= 5;
        }
        float reward_update = reward_increment * invEpisodeLength;
        // print target point x,y, agent_id and reward amount

        //         if(is_dyn_point && (!target_coverage))
        //         {
        //           printf("Emergency %d Pos: %f, %f\n", target_idx, target_x, target_y);
        //           printf("Agent Pos: %f, %f\n", agent_x_arr[kThisEnvAgentsOffset + nearest_agent_id], agent_y_arr[kThisEnvAgentsOffset + nearest_agent_id]);
        //           printf("dist: %f\n", min_dist);
        //         }
        bool dyn_point_covered = is_dyn_point && (target_coverage || min_dist <= kDroneSensingRange / 2 && nearest_agent_id != -1);
        bool regular_point_covered = !is_dyn_point && (min_dist <= kDroneSensingRange && nearest_agent_id != -1);
        if (dyn_point_covered || regular_point_covered) {
          // Covered Emergency or Covered Surveillance
          bool is_drone = agent_types_arr[nearest_agent_id];
          if (!is_dyn_point) {
            // Only Surveillance Points have AoI reset.
            target_aoi = 1;
          } else {
            //               emergency_cover_num++;
            // clear this emergency point in the allocation this_emergency_allocation_table
            for (int i = 0; i < kNumAgents; i++) {
              if (this_emergency_allocation_table[i] == target_idx) {
                this_emergency_allocation_table[i] = -1;
                break;
              }
            }

          }
          // Reward is one time for emergency
          if (!(is_dyn_point && target_coverage)) {
//           printf("Target %d Pos: %f, %f, AoI: %d agent %d receives reward %f\n", target_idx, target_x, target_y,
//           target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx], nearest_agent_id, reward_update);
            rewards_arr[kThisEnvAgentsOffset + nearest_agent_id] += reward_update;
            if (is_drone && !single_type_agent) {
              int drone_nearest_car_id = neighbor_agent_ids_arr[kThisEnvAgentsOffset + nearest_agent_id];
              rewards_arr[kThisEnvAgentsOffset + drone_nearest_car_id] += reward_update;
            }
            global_reward += reward_update;
            target_coverage = true;
            if(is_dyn_point){
//               printf("%d: emergency %d at %f,%f in env %d handled by %d, aoi=%d\n",
//               env_timestep, target_idx, target_x, target_y, kEnvId, nearest_agent_id, target_aoi);
            }
          }
          //             count++;
          //             printf("target %d covered, coverage arr %d\n", target_idx, target_coverage_arr[kThisTargetAgeArrayIdxOffset + target_idx]);
        } else {
          // Uncovered Emergency and Uncovered Surveillance, both require AoI increasing.
          // Note Emergency Points Before Schedule are skipped in prior logic.
          target_aoi++;
          if (is_dyn_point) {
            // scan the this_emergency_allocation_table and confirm this point is not allocated
            int is_allocated = false;
            for (int i = 0; i < kNumAgents; i++) {
              if (this_emergency_allocation_table[i] == target_idx) {
                is_allocated = true;
                break;
              }
            }
            if (!is_allocated) {
              // find the last timestep in aoi_schedule that is smaller than env_timestep
              int first_schedule = 0;
              for (int i = 0; i < emergency_count; i += emergency_per_gen) {
                if (env_timestep < aoi_schedule[i]) {
                  break;
                }
                else{
                  first_schedule = i;
                }
              }
//               printf("Now selecting emergencies for interval > %d\n", aoi_schedule[first_schedule]);
              // collect agents own selection of emergency targets from action_indices_arr
              for(int i = 0;i < kNumAgents;i++){
                int kThisAgentActionIdxOffset = (kThisEnvAgentsOffset + i) * kNumActionDim;
                int agent_selection = action_indices_arr[kThisAgentActionIdxOffset + 1];
                if(agent_selection < emergency_per_gen &&
                agent_selection + first_schedule + zero_shot_start == target_idx){
//                 printf("Agent %d in %d selected emergency %d\n", i, kEnvId, target_idx);
                  this_emergency_allocation_table[i] = target_idx;
                  break;
                }
              }
            }
          }
          global_reward -= is_dyn_point ? 5 * invEpisodeLength : invEpisodeLength;
        }
        //             if (target_idx < 5){
        //               printf("%p Offset: %d Idx: %d\n", target_coverage_arr + kThisTargetAgeArrayIdxOffset + target_idx, kThisTargetAgeArrayIdxOffset, target_idx);
        //               printf("target %d not covered, coverage arr %d\n", target_idx, target_coverage_arr[kThisTargetAgeArrayIdxOffset + target_idx]);
        //             }
        target_aoi_arr[kThisTargetAgeArrayIdxOffset + target_idx] = target_aoi;
        //         if(is_dyn_point){
        //           printf("Emergency %d AoI: %d\n", target_idx, target_aoi);
        //         }
        target_coverage_arr[kThisTargetAgeArrayIdxOffset + target_idx] = target_coverage;
      }
      global_rewards_arr[kEnvId] = global_reward;
      //     if (dynamic_zero_shot){
      //     float factor = 1;
      //     if(valid_emergency_count){
      //       factor = emergency_cover_num * 1.0 / valid_emergency_count;
      //     }
      // //       printf("Env %d Emergency Coverage: %f\n", kEnvId, factor);
      //       global_rewards_arr[kEnvId] *= factor;
      //       // discount decentralized rewards_arr
      //       for (int agent_idx = 0; agent_idx < kNumAgents; agent_idx++) {
      //         rewards_arr[kThisEnvAgentsOffset + agent_idx] *= factor;
      //       }
      //     }

        for(int i = zero_shot_start;i < kNumTargets;i++){
          mean_emergency_aoi += (target_aoi_arr[kThisTargetAgeArrayIdxOffset + i] - 1);
        }
        mean_emergency_aoi /= emergency_count;
    }
    __sync_env_threads(); // Make sure all agents have calculated the reward
    // check emergency allocation and give extra reward
    if (dynamic_zero_shot && kThisAgentId < kNumAgents) {
    bool current_coverage = target_coverage_arr[kThisTargetAgeArrayIdxOffset + this_emergency_allocation_table[kThisAgentId]];
      if (this_emergency_allocation_table[kThisAgentId] != -1 && (!current_coverage)) {
        int emergency_allocated = this_emergency_allocation_table[kThisAgentId];
        // reward divide by delay
//         rewards_arr[kThisAgentArrayIdx] /= target_aoi_arr[kThisTargetAgeArrayIdxOffset + emergency_allocated];
//         printf("Reward of agent %d discounted by %f, now %f\n", kThisAgentId,
//         1.0 / target_aoi_arr[kThisTargetAgeArrayIdxOffset + emergency_allocated],
//         rewards_arr[kThisAgentArrayIdx]);
        float agent_x = agent_x_arr[kThisAgentArrayIdx];
        float agent_y = agent_y_arr[kThisAgentArrayIdx];
        float target_x = target_x_time_list[kThisTargetPositionTimeListIdxOffset + emergency_allocated];
        float target_y = target_y_time_list[kThisTargetPositionTimeListIdxOffset + emergency_allocated];
        float delta_x = (agent_x - target_x) / kAgentXRange;
        float delta_y = (agent_y - target_y) / kAgentYRange;
        // test, calculate bonus as exp^(-3 * dist)
        rewards_arr[kThisAgentArrayIdx] += 5 * exp(-3 * sqrt(delta_x * delta_x + delta_y * delta_y));
//         rewards_arr[kThisAgentArrayIdx] -= sqrt(delta_x * delta_x + delta_y * delta_y) * target_aoi_arr[kThisTargetAgeArrayIdxOffset + emergency_allocated];
//         printf("Distance penalty of %d: %f\n", kThisAgentId, -sqrt(delta_x * delta_x + delta_y * delta_y));
        // print agent, emergency allocated and distance
//         printf("Agent %d in %d allocated to emergency %d, distance: %f\n", kThisAgentId, kEnvId, emergency_allocated,
//         sqrt(delta_x * delta_x + delta_y * delta_y));
      }
      else{
//         printf("Agent %d in %d not allocated to any emergency\n", kThisAgentId, kEnvId);
        rewards_arr[kThisAgentArrayIdx] -= mean_emergency_aoi;
      }
    }
    __sync_env_threads(); // Make sure all agents have calculated the reward
    // Generate State (only the first agent can generate state AoI)
    if (kThisAgentId == 0) {
      // const int global_range = kDroneCarCommRange * 4;
      //       printf("StateAoIGen: %d %p %p\n", kEnvId, state_arr + kThisEnvStateOffset + state_vec_features,
      //       state_arr + kThisEnvStateOffset + state_vec_features + 100);
      memset(state_arr + kThisEnvStateOffset, 0, state_vec_features * sizeof(float));
      //       printf("Grid Center (%f, %f)\n", max_distance_x / 2, max_distance_y / 2);
      CUDACrowdSimGenerateAoIGrid(
        state_arr + kThisEnvStateOffset + state_vec_features,
        max_distance_x >> 1,
        max_distance_y >> 1,
        max_distance_x,
        max_distance_y,
        target_x_time_list + kThisTargetPositionTimeListIdxOffset,
        target_y_time_list + kThisTargetPositionTimeListIdxOffset,
        aoi_schedule,
        target_aoi_arr + kThisTargetAgeArrayIdxOffset,
        kNumTargets,
        kEpisodeLength,
        dynamic_zero_shot,
        zero_shot_start,
        env_timestep,
        kThisAgentId,
        kEnvId
      );
    }
    __sync_env_threads(); // Wait here until state AoI are generated (emergency AoIs are shared.)
    // -------------------------------
    // Compute Observation
    //     printf("GenObs: %d %d\n", kEnvId, kThisAgentId);
    if (kThisAgentId < kNumAgents) {
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
        aoi_schedule,
        target_aoi_arr,
        emergency_index,
        emergency_dis,
        target_coverage_arr,
        grid_flatten_size,
        neighbor_agent_distances_arr,
        neighbor_agent_ids_sorted_by_distances_arr,
        kDroneCarCommRange,
        env_timestep,
        kNumAgents,
        kEpisodeLength,
        obs_features,
        obs_vec_features,
        kEnvId,
        kThisAgentId,
        kThisAgentArrayIdx,
        AgentFeature,
        kThisEnvAgentsOffset,
        kThisEnvStateOffset,
        state_vec_features,
        max_distance_x,
        max_distance_y,
        dynamic_zero_shot,
        zero_shot_start,
        emergency_count,
        FeaturesInEmergencyQueue,
        EmergencyQueueLength
      );
    }

    __sync_env_threads(); // Wait here to update observation before determining done_arr

    // additional reward logic after observation generation
    if (kThisAgentId < kNumAgents) {
    float my_x = agent_x_arr[kThisAgentArrayIdx];
    float my_y = agent_y_arr[kThisAgentArrayIdx];
    float * my_obs_at_emergency = obs_arr + kThisAgentArrayIdx * obs_features + AgentFeature + (kNumAgentsObserved << 2);
    int my_emergency_target = emergency_allocation_table[kThisAgentArrayIdx];
//     for(int i = 0;i < FeaturesInEmergencyQueue * emergency_per_gen;i++){
//       my_obs_at_emergency[i] = 0.0;
//     }
    if(my_emergency_target != -1){
      float target_x = target_x_time_list[kThisTargetPositionTimeListIdxOffset + my_emergency_target];
      float target_y = target_y_time_list[kThisTargetPositionTimeListIdxOffset + my_emergency_target];
      my_obs_at_emergency[0] = target_x / kAgentXRange;
      my_obs_at_emergency[1] = target_y / kAgentYRange;
      my_obs_at_emergency[2] =
      target_aoi_arr[kThisTargetAgeArrayIdxOffset + my_emergency_target] * invEpisodeLength;
      float delta_x = (my_x - target_x) / kAgentXRange;
      float delta_y = (my_y - target_y) / kAgentYRange;
      my_obs_at_emergency[3] =
      sqrt(delta_x * delta_x + delta_y * delta_y) * target_aoi_arr[kThisTargetAgeArrayIdxOffset + my_emergency_target];
      // print four information in a row
//       printf("Agent %d in %d allocated to emergency %d, distance: %f\n", kThisAgentId, kEnvId, my_emergency_target,
//       sqrt(delta_x * delta_x + delta_y * delta_y));
    }
    else{
      for(int i = 0;i < FeaturesInEmergencyQueue;i++){
        my_obs_at_emergency[i] = 0.0;
      }
    }
        // find current generation time
//     int emergency_start_index = -1;
//     for(int i = 0;i < emergency_count; i += emergency_per_gen){
//       if(env_timestep > aoi_schedule[i]){
//         emergency_start_index = i;
//       }
//       else{
//       break;
//       }
//     }
//     // Test, give current emergency points to all agents
//         for(int i = 1;i < emergency_per_gen + 1;i++){
//           float target_x = target_x_time_list[kThisTargetPositionTimeListIdxOffset + emergency_start_index + zero_shot_start];
//           float target_y = target_y_time_list[kThisTargetPositionTimeListIdxOffset + emergency_start_index + zero_shot_start];
//           my_obs_at_emergency[i * FeaturesInEmergencyQueue + 0] = target_x / kAgentXRange;
//           my_obs_at_emergency[i * FeaturesInEmergencyQueue + 1] = target_y / kAgentYRange;
//           my_obs_at_emergency[i * FeaturesInEmergencyQueue + 2] = target_aoi_arr[kThisTargetAgeArrayIdxOffset + emergency_start_index + zero_shot_start] * invEpisodeLength;
//           float delta_x = (my_x - target_x) / kAgentXRange;
//           float delta_y = (my_y - target_y) / kAgentYRange;
//           my_obs_at_emergency[i * FeaturesInEmergencyQueue + 3] = sqrt(delta_x * delta_x + delta_y * delta_y) *
//           target_aoi_arr[kThisTargetAgeArrayIdxOffset + emergency_start_index + zero_shot_start];
//           emergency_start_index++;
//         }
      // energy penalty
      if (agent_energy_arr[kThisAgentArrayIdx] <= 0) {
        rewards_arr[kThisAgentArrayIdx] -= 10;
      }
    }
    // -------------------------------
    // Use only agent 0's thread to set done_arr
    if (kThisAgentId == 0) {
      // State Emergency TODO
      // emergency_queue = state_arr[kThisEnvStateOffset + StateFullAgentFeature];
      bool no_energy = false;
      // run for loop for agents and check agent_energy_arr
      for (int agent_idx = 0; agent_idx < kNumAgents; agent_idx++) {
        if (agent_energy_arr[kThisEnvAgentsOffset + agent_idx] <= 0) {
          no_energy = true;
          break;
        }
      }
      // run for loop for agents_over_range and check over_range status
      if (no_energy) {
        // premature ending should be paired with maximum negative reward
        global_rewards_arr[kEnvId] = -kNumTargets * invEpisodeLength;
      }
      if (env_timestep == kEpisodeLength || no_energy) {
        done_arr[kEnvId] = 1;
        //           printf("coverage: %d\n", count);
      }
      //       printf("Global Reward at %d: %f\n", kEnvId, global_rewards_arr[kEnvId]);
    }
  }
}