#pragma once

#include <arpa/inet.h>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <dirent.h>
#include <fstream>
#include <getopt.h>
#include <ifaddrs.h>
#include <infiniband/verbs.h>
#include <iostream>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include <netinet/ether.h>
#include <netinet/ip.h>
#include <netinet/udp.h>

#include <pcap.h>
#include <sys/time.h> // Include for gettimeofday()

#include "spatial/logging.hpp"
#include "spatial/spatial.hpp"

class LibibverbsPacketCapture : public PacketInput {
private:
  /**
   * Function to create and attach a default flow to the QP.
   * This flow matches all packets and is used as a fallback.
   * @param qp Pointer to the QP to attach the flow to.
   * @return Pointer to the created flow.
   */
  struct ibv_flow *create_and_attach_default_flow(ibv_qp *qp) {
    struct ibv_flow_attr *flow_attr;
    struct ibv_flow_spec_eth *eth_spec;
    void *buf;

    buf = calloc(1, sizeof(*flow_attr) + sizeof(*eth_spec));
    flow_attr = (struct ibv_flow_attr *)buf;
    eth_spec = (struct ibv_flow_spec_eth *)(flow_attr + 1);

    flow_attr->type = IBV_FLOW_ATTR_NORMAL;
    flow_attr->size = sizeof(*flow_attr) + sizeof(*eth_spec);
    flow_attr->priority = 0;
    flow_attr->num_of_specs = 0; // Should be 1.
    flow_attr->port = 1;
    flow_attr->flags = 0;

    eth_spec->type = IBV_FLOW_SPEC_ETH;
    eth_spec->size = sizeof(*eth_spec);
    memset(&eth_spec->val, 0, sizeof(eth_spec->val));   // Match all
    memset(&eth_spec->mask, 0, sizeof(eth_spec->mask)); // No filtering

    struct ibv_flow *flow = ibv_create_flow(qp, flow_attr);
    if (!flow) {
      std::cerr << "Failed to create flow: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      free(buf);
      exit(1);
    }
    return flow;
  }
  /**
   * ]Function to create a flow that matches UDP packets with a specific
   * destination port.
   * @param qp Pointer to the QP to attach the flow to.
   * @param udp_dport The destination port to match.
   */
  struct ibv_flow *create_udp_flow(ibv_qp *qp, uint16_t udp_dport,
                                   uint32_t src_ip = 0) {
    // Allocate memory for flow attributes and specs
    size_t flow_size = sizeof(struct ibv_flow_attr) +
                       sizeof(struct ibv_flow_spec_eth) +
                       sizeof(struct ibv_flow_spec_ipv4) +
                       sizeof(struct ibv_flow_spec_tcp_udp);
    void *flow_mem = calloc(1, flow_size);

    struct ibv_flow_attr *flow_attr = (struct ibv_flow_attr *)flow_mem;
    struct ibv_flow_spec_eth *eth_spec =
        (struct ibv_flow_spec_eth *)(flow_attr + 1);
    struct ibv_flow_spec_ipv4 *ipv4_spec =
        (struct ibv_flow_spec_ipv4 *)(eth_spec + 1);
    struct ibv_flow_spec_tcp_udp *udp_spec =
        (struct ibv_flow_spec_tcp_udp *)(ipv4_spec + 1);

    // Configure flow attributes
    flow_attr->type = IBV_FLOW_ATTR_NORMAL;
    flow_attr->size = flow_size;
    flow_attr->priority = 0;
    flow_attr->num_of_specs = 3; // ETH + IPv4 + UDP
    flow_attr->port = 1;         // Port number (depends on your setup)
    flow_attr->flags = 0;

    // Configure Ethernet spec (match all Ethernet frames)
    eth_spec->type = IBV_FLOW_SPEC_ETH;
    eth_spec->size = sizeof(struct ibv_flow_spec_eth);
    memset(&eth_spec->val, 0, sizeof(eth_spec->val));   // Match all
    memset(&eth_spec->mask, 0, sizeof(eth_spec->mask)); // No filtering

    // Configure IPv4 spec (match all IPv4 packets)
    ipv4_spec->type = IBV_FLOW_SPEC_IPV4;
    ipv4_spec->size = sizeof(struct ibv_flow_spec_ipv4);
    memset(&ipv4_spec->val, 0, sizeof(ipv4_spec->val));   // Match all
    memset(&ipv4_spec->mask, 0, sizeof(ipv4_spec->mask)); // No filtering
    if (src_ip != 0) {
      ipv4_spec->val.src_ip = htonl(src_ip); // match given source IP
      ipv4_spec->mask.src_ip = 0xFFFFFFFF;   // Filter on source IP
      struct in_addr ip_addr;
      ip_addr.s_addr = ntohl(src_ip);
      std::cout << "IP Address: " << inet_ntoa(ip_addr) << std::endl;
    }

    // Configure UDP spec (filter by destination port)
    udp_spec->type = IBV_FLOW_SPEC_UDP;
    udp_spec->size = sizeof(struct ibv_flow_spec_tcp_udp);
    udp_spec->val.dst_port = htons(udp_dport); // Destination port to match
    udp_spec->mask.dst_port = 0xFFFF; // Exact match on destination port
    udp_spec->val.src_port = 0;       // Match all source ports
    udp_spec->mask.src_port = 0;      // No filtering on source port

    // Attach the flow to the QP
    struct ibv_flow *flow = ibv_create_flow(qp, flow_attr);
    if (!flow) {
      std::cerr << "Failed to create flow: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      free(flow_mem);
      return nullptr;
    }

    free(flow_mem);
    return flow;
  }
  // Function to initialize the context and QP
  ibv_qp *init_qp(ibv_context *context, ibv_pd *pd, ibv_cq *cq) {
    ibv_qp_init_attr qp_init_attr = {};
    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.cap.max_send_wr = 1;
    qp_init_attr.cap.max_recv_wr = num_frames;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 2;
    qp_init_attr.qp_type = IBV_QPT_RAW_PACKET;

    ibv_qp *qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
      std::cerr << "Failed to create QP: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      exit(1);
    }

    ibv_qp_attr attr = {};
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = 1;
    // attr.pkey_index = 0; // don't do for raw packet - g et invalid argument
    // attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE | IBV_QP_PORT)) {
      std::cerr << "Failed to modify QP to INIT: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      exit(1);
    }

    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
      std::cerr << "Failed to modify QP to RTR: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      exit(1);
    }

    attr.qp_state = IBV_QPS_RTS;
    if (ibv_modify_qp(qp, &attr, IBV_QP_STATE)) {
      std::cerr << "Failed to modify QP to RTS: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      exit(1);
    }

    // create_and_attach_default_flow(qp);

    return qp;
  }

  void ibname_to_ethname(const char *ibname, char *ethname) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/device/net", ibname);

    DIR *dir = opendir(path);
    if (!dir) {
      perror("opendir");
      return;
    }

    struct dirent *entry;
    // printf("Network interfaces for RDMA device %s:\n",  ibname);
    while ((entry = readdir(dir)) != NULL) {
      if (entry->d_type == DT_DIR && strcmp(entry->d_name, ".") != 0 &&
          strcmp(entry->d_name, "..") != 0) {
        printf("  %s\n", entry->d_name);
        strcpy(ethname, entry->d_name);
        return;
      }
    }

    closedir(dir);
  }

  void get_interface_ip(const char *interface_name, struct sockaddr_in *addr) {
    struct ifaddrs *ifaddr;
    if (getifaddrs(&ifaddr) == -1) {
      std::cerr << "Failed to get network interfaces: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      return;
    }

    for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
      if (ifa->ifa_addr == nullptr ||
          strcmp(ifa->ifa_name, interface_name) != 0) {
        continue;
      }

      if (ifa->ifa_addr->sa_family == AF_INET) { // IPv4
        char ip[INET_ADDRSTRLEN];
        *addr = *(struct sockaddr_in *)ifa->ifa_addr;
        inet_ntop(AF_INET, &addr->sin_addr, ip, INET_ADDRSTRLEN);
        // std::cout << "Interface: " << interface_name << ", IP Address: " <<
        // ip << std::endl;
        freeifaddrs(ifaddr);
        return;
      }
    }

    std::cerr << "No IPv4 address found for interface: " << interface_name
              << std::endl;
    freeifaddrs(ifaddr);
    addr = nullptr;
  }

  void subscribe_to_multicast(const char *interface_name, ibv_qp *qp,
                              const char *multicast_ip, uint16_t udp_port) {

    // Get IP
    char ethname[256];
    ibname_to_ethname(interface_name, ethname);
    // std::cout << "Device " << interface_name << " " << ethname << " " <<
    // std::endl;
    struct sockaddr_in ipaddr;
    get_interface_ip(ethname, &ipaddr);

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
      std::cerr << "Failed to create socket: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      return;
    }

    // Bind the socket to the specified port
    struct sockaddr_in local_addr = {};
    local_addr.sin_family = AF_INET;
    local_addr.sin_port = htons(udp_port);
    local_addr.sin_addr.s_addr =
        htonl(INADDR_ANY); // Bind to all local interfaces

    if (bind(sock, (struct sockaddr *)&local_addr, sizeof(local_addr)) < 0) {
      std::cerr << "Failed to bind socket to port " << udp_port << ": "
                << strerror(errno) << " (errno: " << errno << ")" << std::endl;
      close(sock);
      return;
    }

    struct ip_mreq mreq;
    memset(&mreq, 0, sizeof(mreq));

    // Set the multicast group address
    mreq.imr_multiaddr.s_addr = inet_addr(multicast_ip);
    if (mreq.imr_multiaddr.s_addr == INADDR_NONE) {
      std::cerr << "Invalid multicast IP address: " << multicast_ip
                << std::endl;
      close(sock);
      return;
    }

    if (ipaddr.sin_addr.s_addr == 0) {
      std::cerr << "Failed to get IP address for interface: " << interface_name
                << std::endl;
      close(sock);
      return;
    }

    // Set the interface for the multicast group
    mreq.imr_interface.s_addr = ipaddr.sin_addr.s_addr;

    char ip_str[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &mreq.imr_interface.s_addr, ip_str, INET_ADDRSTRLEN);
    std::cout << "Joining multicast group " << multicast_ip << " on interface "
              << ip_str << std::endl;

    // Subscribe to the multicast group
    if (setsockopt(sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq, sizeof(mreq)) <
        0) {
      std::cerr << "Failed to join multicast group: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      close(sock);
      return;
    }

    std::cout << "Successfully subscribed " << interface_name
              << " to multicast group " << multicast_ip << " on port "
              << udp_port << std::endl;

    // Close the socket
    // close(sock);
  }
  void post_recv(ibv_qp *qp, ibv_mr *header_mr, ibv_mr *mr, char *header_buffer,
                 char *data_buffer, uint32_t frame_id, uint32_t qpid,
                 bool separate_header) {
    ibv_sge sge[2] = {}; // Array to hold up to 2 SGEs

    if (separate_header) {
      // First SGE: Headers in the header buffer
      sge[0].addr = reinterpret_cast<uintptr_t>(header_buffer + frame_id * MTU);
      sge[0].length = TOTAL_HDR_SIZE; // Length of the headers
      sge[0].lkey = header_mr->lkey;

      // Second SGE: Rest of the packet in the data buffer
      sge[1].addr = reinterpret_cast<uintptr_t>(data_buffer + frame_id * MTU);
      sge[1].length = MTU - TOTAL_HDR_SIZE; // Remaining packet size
      sge[1].lkey = mr->lkey;
    } else {
      // Single SGE: Entire packet in the data buffer
      sge[0].addr = reinterpret_cast<uintptr_t>(data_buffer + frame_id * MTU);
      sge[0].length = MTU; // Full packet size
      sge[0].lkey = mr->lkey;
    }

    ibv_recv_wr wr = {};
    wr.wr_id = static_cast<uint64_t>(qpid) << 32 |
               frame_id; // Encode QP ID and frame ID
    wr.sg_list = sge;
    wr.num_sge = separate_header
                     ? 2
                     : 1; // Use 2 SGEs if save_gpu is enabled, otherwise 1

    ibv_recv_wr *bad_wr;
    if (ibv_post_recv(qp, &wr, &bad_wr)) {
      std::cerr << "Failed to post receive work request: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      exit(1);
    }
  }

public:
  LibibverbsPacketCapture() { print_device_list(); };
  ~LibibverbsPacketCapture();

  void get_packets(ProcessorStateBase &state) override {

  };

  void print_device_list() {
    int num_devices = 0;
    // Get the list of devices
    ibv_device **device_list = ibv_get_device_list(&num_devices);
    if (!device_list) {
      std::cerr << "Failed to get IB devices list: " << strerror(errno)
                << " (errno: " << errno << ")" << std::endl;
      exit(-1);
    }

    // Print the list of devices
    std::cout << "Available devices:" << std::endl;
    for (int i = 0; i < num_devices; ++i) {
      interfaces[i] = ibv_get_device_name(device_list[i]);
      char ethname[256];
      ibname_to_ethname(interfaces[i], ethname);
      std::cout << "Device " << i << ": " << interfaces[i] << " " << ethname
                << " " << std::endl;
      struct sockaddr_in ipaddr;
      get_interface_ip(ethname, &ipaddr);
    }

    // Free the device list
    ibv_free_device_list(device_list);
  }
}
