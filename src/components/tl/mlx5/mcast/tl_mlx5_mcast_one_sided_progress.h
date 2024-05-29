/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "tl_mlx5_mcast.h"
#include "tl_mlx5_mcast_helper.h"
#include "p2p/ucc_tl_mlx5_mcast_p2p.h"

#ifndef TL_MLX5_MCAST_ONE_SIDED_PROGRESS_H_
#define TL_MLX5_MCAST_ONE_SIDED_PROGRESS_H_

ucc_status_t ucc_tl_mlx5_mcast_progress_one_sided_communication(mcast_coll_comm_t *comm, mcast_coll_req_t *req);

ucc_status_t ucc_tl_mlx5_mcast_reliable_one_sided_get(mcast_coll_comm_t *comm, mcast_coll_req_t *req, int *completed);

ucc_status_t ucc_tl_mlx5_mcast_reliable_zcopy_pipelined_one_sided_get(mcast_coll_comm_t *comm,
                                                                      mcast_coll_req_t *req,
                                                                      int *completed);

ucc_status_t ucc_tl_mlx5_mcast_process_packet_collective(mcast_coll_comm_t *comm,
                                                         mcast_coll_req_t *req,
                                                         struct pp_packet* pp, int coll_type);

#endif

