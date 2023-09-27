/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <inttypes.h>
#include "tl_mlx5_mcast.h"
#include "tl_mlx5_mcast_helper.h"
#include "tl_mlx5_mcast_rcache.h"
#include "utils/arch/cpu.h"
#include <ucs/sys/string.h>
#include "core/ucc_service_coll.h"
#include "tl_mlx5.h"

ucc_status_t ucc_tl_mlx5_mcast_context_init(ucc_tl_mlx5_mcast_context_t    *context,
                                            ucc_tl_mlx5_mcast_ctx_params_t *mcast_ctx_conf)
{
    ucc_status_t            status      = UCC_OK;
    struct ibv_device     **device_list = NULL;
    struct ibv_device      *dev         = NULL;
    char                   *devname     = NULL;
    int                     is_ipv4     = 0;
    struct sockaddr_in     *in_src_addr = NULL;
    struct rdma_cm_event   *revent      = NULL;
    char                   *ib          = NULL;
    char                   *ib_name     = NULL;
    char                   *port        = NULL;
    struct ibv_port_attr    port_attr;
    struct ibv_device_attr  device_attr;
    struct sockaddr_storage ip_oib_addr;
    struct sockaddr_storage dst_addr;
    int                     num_devices;
    int                     active_mtu;
    int                     max_mtu;
    char                    addrstr[128];
    ucc_tl_mlx5_context_t  *mlx5_ctx;
    ucc_base_lib_t         *lib;

    ucc_tl_mlx5_mcast_coll_context_t *ctx = NULL;

    ctx = &(context->mcast_context);
    memset(ctx, 0, sizeof(ucc_tl_mlx5_mcast_coll_context_t));
    memcpy(&ctx->params, mcast_ctx_conf, sizeof(ucc_tl_mlx5_mcast_ctx_params_t));

    mlx5_ctx = ucc_container_of(context, ucc_tl_mlx5_context_t, mcast);
    lib      = mlx5_ctx->super.super.lib;
    ctx->lib = lib;

    device_list = ibv_get_device_list(&num_devices);
    if (!device_list) {
        tl_error(lib, "no ib devices available");
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    if (num_devices >= 1 && !strcmp(mcast_ctx_conf->ib_dev_name, "")) {
        dev          = device_list[0];
        devname      = (char *)ibv_get_device_name(dev);
        ctx->devname = ucc_malloc(strlen(devname)+16, "devname");
        if (!ctx->devname) {
            status = UCC_ERR_NO_MEMORY;
            goto error;
        }
        strcpy(ctx->devname, devname);
        strcat(ctx->devname, ":1");
    } else {
        ctx->devname = mcast_ctx_conf->ib_dev_name;
    }

    ibv_free_device_list(device_list);
    if (UCC_OK != ucc_tl_probe_ip_over_ib(ctx->devname, &ip_oib_addr)) {
        tl_error(lib, "failed to get ipoib interface for devname %s", ctx->devname);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    is_ipv4     = (ip_oib_addr.ss_family == AF_INET) ? 1 : 0;
    in_src_addr = (struct sockaddr_in*)&ip_oib_addr;

    inet_ntop((is_ipv4) ? AF_INET : AF_INET6,
              &in_src_addr->sin_addr, addrstr, sizeof(addrstr) - 1);
    tl_debug(ctx->lib, "devname %s, ipoib %s", ctx->devname, addrstr);

    ctx->channel = rdma_create_event_channel();
    if (!ctx->channel) {
        tl_error(lib, "rdma_create_event_channel failed, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    memset(&dst_addr, 0, sizeof(struct sockaddr_storage));
    dst_addr.ss_family = is_ipv4 ? AF_INET : AF_INET6;
    if (rdma_create_id(ctx->channel, &ctx->id, NULL, RDMA_PS_UDP)) {
        tl_error(lib, "failed to create rdma id, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    if (0 != rdma_resolve_addr(ctx->id, (struct sockaddr *)&ip_oib_addr,
                (struct sockaddr *) &dst_addr, 1000)) {
        tl_error(lib, "failed to resolve rdma addr, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    if (rdma_get_cm_event(ctx->channel, &revent) < 0 ||
        revent->event != RDMA_CM_EVENT_ADDR_RESOLVED) {
        tl_error(lib, "failed to get cm event, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    rdma_ack_cm_event(revent);
    ctx->ctx = ctx->id->verbs;
    ctx->pd  = ibv_alloc_pd(ctx->ctx);
    if (!ctx->pd) {
        tl_error(lib, "failed to allocate pd");
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    ib           = strdup(ctx->devname);
    ucs_string_split(ib, ":", 2, &ib_name, &port);
    ctx->ib_port = atoi(port);

    /* Determine MTU */
    if (ibv_query_port(ctx->ctx, ctx->ib_port, &port_attr)) {
        tl_error(lib, "couldn't query port in ctx create, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    if (port_attr.max_mtu == IBV_MTU_256)
        max_mtu = 256;
    if (port_attr.max_mtu == IBV_MTU_512)
        max_mtu = 512;
    if (port_attr.max_mtu == IBV_MTU_1024)
        max_mtu = 1024;
    if (port_attr.max_mtu == IBV_MTU_2048)
        max_mtu = 2048;
    if (port_attr.max_mtu == IBV_MTU_4096)
        max_mtu = 4096;

    if (port_attr.active_mtu == IBV_MTU_256)
        active_mtu = 256;
    if (port_attr.active_mtu == IBV_MTU_512)
        active_mtu = 512;
    if (port_attr.active_mtu == IBV_MTU_1024)
        active_mtu = 1024;
    if (port_attr.active_mtu == IBV_MTU_2048)
        active_mtu = 2048;
    if (port_attr.active_mtu == IBV_MTU_4096)
        active_mtu = 4096;

    ctx->mtu = active_mtu;

    if (port_attr.max_mtu < port_attr.active_mtu) {
        tl_debug(ctx->lib, "port active MTU (%d) is smaller than port max MTU (%d)",
                    active_mtu, max_mtu);
    }
    if (ibv_query_device(ctx->ctx, &device_attr)) {
        tl_error(lib, "failed to query device in ctx create, errno %d", errno);
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    tl_debug(ctx->lib, "MTU %d, MAX QP WR: %d, max sqr_wr: %d, max cq: %d, max cqe: %d",
                ctx->mtu, device_attr.max_qp_wr, device_attr.max_srq_wr,
                device_attr.max_cq, device_attr.max_cqe);

    ctx->max_qp_wr = device_attr.max_qp_wr;
    {
        status = ucc_mpool_init(&ctx->compl_objects_mp, 0, sizeof(ucc_tl_mlx5_mcast_p2p_completion_obj_t), 0,
                                UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                                &ucc_coll_task_mpool_ops,
                                UCC_THREAD_SINGLE,
                                "ucc_tl_mlx5_mcast_p2p_completion_obj_t");
        if (ucc_unlikely(UCC_OK != status)) {
            tl_error(lib, "failed to initialize compl_objects_mp mpool");
            status = UCC_ERR_NO_MEMORY;
            goto error;
        }
    }

    ctx->rcache = NULL;
    if (UCC_OK != ucc_tl_mlx5_mcast_setup_rcache(ctx)) {
        tl_error(lib, "failed to setup rcache");
        status = UCC_ERR_NO_RESOURCE;
        goto error;
    }

    tl_debug(ctx->lib, "tl mcast setup complete: ctx %p", ctx);

    return UCC_OK;

error:
    if (ctx->pd) {
        ibv_dealloc_pd(ctx->pd);
    }
    if (ctx->id) {
        rdma_destroy_id(ctx->id);
    }
    if (ctx->channel) {
        rdma_destroy_event_channel(ctx->channel);
    }
    return status;
}
