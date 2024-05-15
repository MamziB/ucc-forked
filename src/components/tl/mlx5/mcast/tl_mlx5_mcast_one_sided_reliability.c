/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_one_sided_reliability.h"

static ucc_status_t ucc_tl_mlx5_mcast_one_sided_setup_reliability_buffers(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *tl_team  = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_status_t                   status   = UCC_OK;
    ucc_tl_mlx5_mcast_coll_comm_t *comm     = tl_team->mcast->mcast_comm;
    int                            one_sided_total_slots_size, i;

    /* this array keeps track of the number of recv packets from each process
     * used in all the protocols */
    comm->recv_list = ucc_calloc(1, comm->commsize * sizeof(uint32_t), "recv_list");
    if (!comm->recv_list) {
        tl_error(comm->lib, "unable to malloc for recv_list");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }

    /* this array holds all the remote-addr/rkey of sendbuf from processes in the team 
     * used in sync design. Needs to be set during each mcast-allgather call after sendbuf 
     * registeration */
    comm->slot_mem_info_list = ucc_calloc(1, comm->commsize * sizeof(ucc_tl_mlx5_mcast_slot_mem_info_t),
                                          "slot_mem_info_list");
    if (!comm->slot_mem_info_list) {
        tl_error(comm->lib, "unable to malloc for slot_mem_info_list");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }

    /* below data structures are used in async design only */
    comm->one_sided_async_slots_size = comm->one_sided_reliability_scheme_msg_threshold
                                           + ONE_SIDED_SLOTS_INFO_SIZE;
    one_sided_total_slots_size       = comm->one_sided_async_slots_size *
                                            ONE_SIDED_SLOTS_COUNT * sizeof(char);
    comm->one_sided_slots_buffer     = (char *)ucc_calloc(1, one_sided_total_slots_size,
                                                          "one_sided_slots_buffer");
    if (!comm->one_sided_slots_buffer) {
        tl_error(comm->lib, "unable to malloc for one_sided_slots_buffer");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }
    comm->one_sided_slots_mr = ibv_reg_mr(comm->ctx->pd, comm->one_sided_slots_buffer,
                                          one_sided_total_slots_size, IBV_ACCESS_LOCAL_WRITE |
                                          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!comm->one_sided_slots_mr) {
        tl_error(comm->lib, "unable to register for one_sided_slots_mr");
        status = UCC_ERR_NO_RESOURCE;
        goto failed;
    }
    
    /* this array holds local information about the slot status that was read from remote ranks */
    comm->remote_slot_info = ucc_calloc(1, comm->commsize * ONE_SIDED_SLOTS_INFO_SIZE,
                                        "remote_slot_info");
    if (!comm->remote_slot_info) {
        tl_error(comm->lib, "unable to malloc for remote_slot_info");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }
    comm->remote_slot_info_mr = ibv_reg_mr(comm->ctx->pd, comm->remote_slot_info,
                                           comm->commsize * ONE_SIDED_SLOTS_INFO_SIZE,
                                           IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                                           IBV_ACCESS_REMOTE_READ);
    if (!comm->remote_slot_info_mr) {
        tl_error(comm->lib, "unable to register for remote_slot_info_mr");
        status = UCC_ERR_NO_RESOURCE;
        goto failed;
    }

    comm->one_sided_reliability_info = ucc_calloc(1, sizeof(ucc_tl_mlx5_one_sided_reliable_team_info_t) *
                                                  comm->commsize, "one_sided_reliability_info");
    if (!comm->one_sided_reliability_info) {
        tl_error(comm->lib, "unable to allocate mem for one_sided_reliability_info");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }

    status = ucc_tl_mlx5_mcast_create_rc_qps(comm->ctx, comm);
    if (UCC_OK != status) {
        tl_error(comm->lib, "RC qp create failed");
        goto failed;
    }

    /* below holds the remote addr/rkey to local slot field of all the
     * processes used in async protocol */
    comm->one_sided_reliability_info[comm->rank].slot_mem.rkey        = comm->one_sided_slots_mr->rkey;
    comm->one_sided_reliability_info[comm->rank].slot_mem.remote_addr = (uint64_t)comm->one_sided_slots_buffer;
    comm->one_sided_reliability_info[comm->rank].port_lid             = comm->ctx->port_lid;
    for (i = 0; i < comm->commsize; i++) {
        comm->one_sided_reliability_info[comm->rank].rc_qp_num[i] = comm->mcast.rc_qp[i]->qp_num;
    }

    tl_debug(comm->lib, "created the allgather reliability structures");

    return UCC_OK;

failed:
    if (comm->one_sided_slots_mr) {
        ibv_dereg_mr(comm->one_sided_slots_mr);
    }

    if (comm->remote_slot_info_mr) {
        ibv_dereg_mr(comm->remote_slot_info_mr);
    }

    if (comm->one_sided_slots_buffer) {
        ucc_free(comm->one_sided_slots_buffer);
    }

    if (comm->recv_list) {
        ucc_free(comm->recv_list);
    }

    if (comm->slot_mem_info_list) {
        ucc_free(comm->slot_mem_info_list);
    }

    if (comm->remote_slot_info) {
        ucc_free(comm->remote_slot_info);
    }

    if (comm->one_sided_reliability_info) {
        ucc_free(comm->one_sided_reliability_info);
    }

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_one_sided_reliability_init(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *tl_team  = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_tl_mlx5_mcast_coll_comm_t *comm     = tl_team->mcast->mcast_comm;
    ucc_status_t                   status   = UCC_OK;

    status = ucc_tl_mlx5_mcast_one_sided_setup_reliability_buffers(team);
    if (status != UCC_OK) {
        tl_error(comm->lib, "setup reliablity buffers failed");
        goto cleanup;
    }

     /* TODO double check if ucc inplace allgather is working properly */
    status = comm->allgather_post(comm->p2p_ctx, NULL /*inplace*/, comm->one_sided_reliability_info,
                                  sizeof(ucc_tl_mlx5_one_sided_reliable_team_info_t),
                                  &comm->one_sided_reliability_req);
    if (UCC_OK != status) {
        goto cleanup;
    }

cleanup:
    return status;

}

ucc_status_t ucc_tl_mlx5_mcast_one_sided_reliability_test(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *tl_team  = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_status_t                   status   = UCC_OK;
    ucc_tl_mlx5_mcast_coll_comm_t *comm     = tl_team->mcast->mcast_comm;

    /* check if the one sided config info is exchanged */
    status = comm->coll_test(comm->one_sided_reliability_req);
    if (UCC_OK != status) {
        /* allgather is not completed yet */
        if (status < 0) {
            tl_error(comm->lib, "one sided config info exchange failed");
            goto failed;
        }
        return status;
    }

    /* we have all the info to make the reliable connections */
    status = ucc_tl_mlx5_mcast_modify_rc_qps(comm->ctx, comm);
    if (UCC_OK != status) {
        tl_error(comm->lib, "RC qp modify failed");
        goto failed;
    }

failed:
    if (comm->one_sided_slots_mr) {
        ibv_dereg_mr(comm->one_sided_slots_mr);
    }

    if (comm->remote_slot_info_mr) {
        ibv_dereg_mr(comm->remote_slot_info_mr);
    }

    if (comm->one_sided_slots_buffer) {
        ucc_free(comm->one_sided_slots_buffer);
    }

    if (comm->recv_list) {
        ucc_free(comm->recv_list);
    }

    if (comm->slot_mem_info_list) {
        ucc_free(comm->slot_mem_info_list);
    }

    if (comm->remote_slot_info) {
        ucc_free(comm->remote_slot_info);
    }

    if (comm->one_sided_reliability_info) {
        ucc_free(comm->one_sided_reliability_info);
    }

    return status;
}
