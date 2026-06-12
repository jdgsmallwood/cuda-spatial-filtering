// fast_udp_sender — rate-controlled LAMBDA-format UDP load generator.
//
// Synthesizes packets in the on-wire format observe expects when receiving on a
// kernel socket (CustomHeader + scales + samples, no eth/ip/udp headers in the
// payload), so no PCAP file is needed.  sendmmsg() batches + absolute-deadline
// pacing reach multi-Mpps per core, unlike apps/udp_sender.cpp (usleep(5) per
// packet caps it well under 0.1 Mpps).
//
// Build:   gcc -O2 -o fast_udp_sender fast_udp_sender.c -lpthread
// Example: ./fast_udp_sender --dst 127.0.0.1:36001 --pps 2000000 --threads 4 \
//              --channels 8 --min-channel 0 --duration 10
//
// Note: observe's LambdaConfig sets OVERWRITE_FPGA_ID_WITH_IP_THIRD_OCTET=true,
// so the receiver derives the FPGA id from the *source IP third octet*.  Use
// --src-ip 10.0.<fpga>.1 (one process per FPGA source) for multi-FPGA tests;
// the --fpga-id header field only matters when that template flag is false.

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#pragma pack(push, 1)
typedef struct {
  uint64_t sample_count;
  uint32_t fpga_id;
  uint16_t freq_channel;
  uint8_t padding[8];
} CustomHeader;
#pragma pack(pop)

#define SEND_BATCH 64

static struct {
  char dst_ip[64];
  int dst_port;
  char src_ip[64]; // optional bind address (third octet -> fpga id)
  long pps;        // aggregate target across threads
  int threads;
  int duration_s;   // 0 = run forever
  int channels;     // freq channels emitted, [min_channel, min_channel+channels)
  int min_channel;
  int receivers;    // NR_RECEIVERS_PER_PACKET
  int timesteps;    // NR_TIME_STEPS_PER_PACKET (also seq increment per packet)
  uint32_t fpga_id; // header field (used when IP-octet override is off)
} cfg = {.dst_ip = "127.0.0.1",
         .dst_port = 36001,
         .src_ip = "",
         .pps = 1000000,
         .threads = 2,
         .duration_s = 10,
         .channels = 8,
         .min_channel = 0,
         .receivers = 10,
         .timesteps = 64,
         .fpga_id = 0};

static atomic_long g_sent = 0;
static atomic_long g_errors = 0;
static atomic_int g_stop = 0;

static int payload_size(void) {
  // CustomHeader + int16 scales[R][2] + complex<int8> samples[T][R][2]
  return (int)sizeof(CustomHeader) + cfg.receivers * 2 * 2 +
         cfg.timesteps * cfg.receivers * 2 * 2;
}

static void ts_add_ns(struct timespec *t, long ns) {
  t->tv_nsec += ns;
  while (t->tv_nsec >= 1000000000L) {
    t->tv_nsec -= 1000000000L;
    t->tv_sec += 1;
  }
}

typedef struct {
  int id;
} thread_arg;

static void *sender_thread(void *argp) {
  thread_arg *arg = (thread_arg *)argp;
  const int psz = payload_size();
  const long per_thread_pps = cfg.pps / cfg.threads;

  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    perror("socket");
    return NULL;
  }
  int sndbuf = 64 * 1024 * 1024;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

  if (cfg.src_ip[0]) {
    struct sockaddr_in src = {.sin_family = AF_INET, .sin_port = 0};
    inet_pton(AF_INET, cfg.src_ip, &src.sin_addr);
    if (bind(fd, (struct sockaddr *)&src, sizeof(src)) < 0)
      perror("bind src (continuing unbound)");
  }

  struct sockaddr_in dst = {.sin_family = AF_INET,
                            .sin_port = htons(cfg.dst_port)};
  inet_pton(AF_INET, cfg.dst_ip, &dst.sin_addr);
  if (connect(fd, (struct sockaddr *)&dst, sizeof(dst)) < 0) {
    perror("connect");
    close(fd);
    return NULL;
  }

  // One reusable buffer per batch slot; headers rewritten each send.
  uint8_t *bufs = calloc(SEND_BATCH, psz);
  struct iovec iov[SEND_BATCH];
  struct mmsghdr msgs[SEND_BATCH];
  memset(msgs, 0, sizeof(msgs));
  for (int i = 0; i < SEND_BATCH; ++i) {
    // Deterministic non-zero sample data (ramp) past the header+scales.
    uint8_t *p = bufs + (size_t)i * psz;
    int16_t *scales = (int16_t *)(p + sizeof(CustomHeader));
    for (int s = 0; s < cfg.receivers * 2; ++s)
      scales[s] = 1;
    uint8_t *data = p + sizeof(CustomHeader) + cfg.receivers * 2 * 2;
    for (int b = 0; b < cfg.timesteps * cfg.receivers * 2 * 2; ++b)
      data[b] = (uint8_t)(b & 0x3f);
    iov[i].iov_base = p;
    iov[i].iov_len = psz;
    msgs[i].msg_hdr.msg_iov = &iov[i];
    msgs[i].msg_hdr.msg_iovlen = 1;
  }

  // Channels are striped across threads; each thread emits full sample_count
  // rows for its channel subset (mimics per-channel streams sharing counts).
  int my_channels[256];
  int n_my = 0;
  for (int c = arg->id; c < cfg.channels && n_my < 256; c += cfg.threads)
    my_channels[n_my++] = cfg.min_channel + c;
  if (n_my == 0) {
    close(fd);
    free(bufs);
    return NULL;
  }

  // sample_count starts non-zero (observe treats 0 as fatal) and advances by
  // `timesteps` per packet row, matching the FPGA convention.
  uint64_t sample_count = (uint64_t)cfg.timesteps * 1000;
  int chan_idx = 0;

  const long batch_ns =
      (long)((double)SEND_BATCH * 1e9 / (double)per_thread_pps);
  struct timespec next;
  clock_gettime(CLOCK_MONOTONIC, &next);

  while (!atomic_load_explicit(&g_stop, memory_order_relaxed)) {
    for (int i = 0; i < SEND_BATCH; ++i) {
      CustomHeader *h = (CustomHeader *)(bufs + (size_t)i * psz);
      h->sample_count = sample_count;
      h->fpga_id = cfg.fpga_id;
      h->freq_channel = (uint16_t)my_channels[chan_idx];
      if (++chan_idx == n_my) {
        chan_idx = 0;
        sample_count += (uint64_t)cfg.timesteps;
      }
    }
    int sent = sendmmsg(fd, msgs, SEND_BATCH, 0);
    if (sent < 0) {
      if (errno == EINTR)
        continue;
      atomic_fetch_add(&g_errors, 1);
      // ENOBUFS etc: brief backoff, keep deadline schedule.
      struct timespec back = {0, 50000};
      nanosleep(&back, NULL);
    } else {
      atomic_fetch_add_explicit(&g_sent, sent, memory_order_relaxed);
    }
    ts_add_ns(&next, batch_ns);
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, NULL);
  }

  close(fd);
  free(bufs);
  return NULL;
}

static void usage(const char *prog) {
  fprintf(stderr,
          "usage: %s [--dst ip:port] [--src-ip ip] [--pps n] [--threads n]\n"
          "          [--duration secs (0=forever)] [--channels n]\n"
          "          [--min-channel n] [--receivers n] [--timesteps n]\n"
          "          [--fpga-id n]\n",
          prog);
  exit(1);
}

int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    char *a = argv[i];
    char *v = (i + 1 < argc) ? argv[i + 1] : NULL;
    if (!strcmp(a, "--dst") && v) {
      char *colon = strchr(v, ':');
      if (colon) {
        *colon = 0;
        cfg.dst_port = atoi(colon + 1);
      }
      snprintf(cfg.dst_ip, sizeof(cfg.dst_ip), "%s", v);
      ++i;
    } else if (!strcmp(a, "--src-ip") && v) {
      snprintf(cfg.src_ip, sizeof(cfg.src_ip), "%s", v);
      ++i;
    } else if (!strcmp(a, "--pps") && v) {
      cfg.pps = atol(v);
      ++i;
    } else if (!strcmp(a, "--threads") && v) {
      cfg.threads = atoi(v);
      ++i;
    } else if (!strcmp(a, "--duration") && v) {
      cfg.duration_s = atoi(v);
      ++i;
    } else if (!strcmp(a, "--channels") && v) {
      cfg.channels = atoi(v);
      ++i;
    } else if (!strcmp(a, "--min-channel") && v) {
      cfg.min_channel = atoi(v);
      ++i;
    } else if (!strcmp(a, "--receivers") && v) {
      cfg.receivers = atoi(v);
      ++i;
    } else if (!strcmp(a, "--timesteps") && v) {
      cfg.timesteps = atoi(v);
      ++i;
    } else if (!strcmp(a, "--fpga-id") && v) {
      cfg.fpga_id = (uint32_t)atoi(v);
      ++i;
    } else {
      usage(argv[0]);
    }
  }
  if (cfg.threads < 1 || cfg.pps < cfg.threads || cfg.channels < 1)
    usage(argv[0]);

  fprintf(stderr,
          "fast_udp_sender: dst=%s:%d pps=%ld threads=%d payload=%dB "
          "channels=[%d,%d) duration=%ds\n",
          cfg.dst_ip, cfg.dst_port, cfg.pps, cfg.threads, payload_size(),
          cfg.min_channel, cfg.min_channel + cfg.channels, cfg.duration_s);

  pthread_t tids[256];
  thread_arg args[256];
  if (cfg.threads > 256)
    cfg.threads = 256;
  for (int t = 0; t < cfg.threads; ++t) {
    args[t].id = t;
    pthread_create(&tids[t], NULL, sender_thread, &args[t]);
  }

  long last = 0;
  for (int s = 0; cfg.duration_s == 0 || s < cfg.duration_s; ++s) {
    sleep(1);
    long now = atomic_load(&g_sent);
    fprintf(stderr, "[%3ds] sent=%ld  rate=%.3f Mpps  errors=%ld\n", s + 1,
            now, (now - last) / 1e6, atomic_load(&g_errors));
    last = now;
  }
  atomic_store(&g_stop, 1);
  for (int t = 0; t < cfg.threads; ++t)
    pthread_join(tids[t], NULL);

  // Final line is machine-parseable for 01_ingest_sweep.sh.
  printf("TOTAL_SENT=%ld\n", atomic_load(&g_sent));
  return 0;
}
