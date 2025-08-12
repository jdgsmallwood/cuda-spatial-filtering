"""
Author: Jay Smallwood
Email: jsmallwood@swin.edu.au

RDMA Packet simulator on GPU.

Assumptions & Decisions:
 - Packets cannot arrive out-of-order.
 - It is possible to drop packets.
 - If packets are dropped, replace missing data with zeroes.
 - Each packet contains data for one channel & all receivers.
 - Have one ring buffer per channel.
 - Sequence numbers for different channels will be aligned on timestamp (i.e. seq 10 is the same time period for all channels)
 - Written in a C++/CUDA style - i.e. don't use any Python-specific collections or functionality.
 - Use a secondary boolean array to keep track of where new packets have been written.
 - Initial base sequence numbers for each ring buffer is set by the first packet that arrives for any ring buffer, to sync up the blocks between channels.

Main Questions:
 - Even though this is using async it is still single-threaded. Are there race conditions to consider in the RDMA context?
 - How to scale this if not all receivers are within each packet (i.e. some subset. Is that the time to go to receiver/channel combination buffers?)
"""

import asyncio
import enum
import random
from typing import Union

NUM_PACKETS_FOR_PROCESSING = 50
NUM_BLOCKS_IN_BUFFER = 5
BUFFER_SIZE = NUM_PACKETS_FOR_PROCESSING * NUM_BLOCKS_IN_BUFFER
BLOCK_SIZE = NUM_PACKETS_FOR_PROCESSING
DROP_PROB = 0.001

NUM_DROPPED_PACKETS = 0


class BlockReadyReason(enum.Enum):
    LAST_PACKET_ARRIVED = "Last packet arrived"
    DROPPED_LAST_PACKET = "Dropped last packet"
    NOT_READY = "Not ready"


class Packet:
    def __init__(self, channel: int, seq: int):
        self.channel = channel
        self.seq = seq

    def __repr__(self):
        return f"Pkt(c={self.channel}, seq={self.seq})"


class RingBuffer:
    def __init__(self, size: int, block_size: int, channel: int):
        self.size = size
        self.data = [None] * size
        self.packet_arrived = [0] * size
        self.head = 0  # next write position
        self.tail = 0  # next read position
        self.current_seq = 0  # current block sequence number
        self.last_arrived_seq = 0
        self.block_size = block_size
        self.channel = channel

    def add(self, pkt: Packet) -> None:
        if self.current_seq == 0:
            self.current_seq = pkt.seq

        if pkt.seq < self.current_seq:
            print(
                f"Dropping packet with seq num {pkt.seq} as it is behind the current block head {self.current_seq} for channel {self.channel}."
            )

        if pkt.seq - self.current_seq > self.size:
            raise ValueError("Buffer has eaten its own tail!")
        next_idx = (self.head + (pkt.seq - self.current_seq)) % self.size
        self.data[next_idx] = pkt
        self.packet_arrived[next_idx] = 1
        self.last_arrived_seq = pkt.seq

    def block_ready(self) -> tuple[bool, str]:
        # If we assume no out-of-order blocks, then arrival of later block means
        # that no more are coming for this block
        if self.last_arrived_seq == self.current_seq + self.block_size:
            return (True, BlockReadyReason.LAST_PACKET_ARRIVED)
        elif self.last_arrived_seq >= self.current_seq + self.block_size:
            return (True, BlockReadyReason.DROPPED_LAST_PACKET)
        return (False, BlockReadyReason.NOT_READY)

    def get_block(
        self,
    ) -> tuple[Union[None, tuple[list[Packet]]], Union[None, list[bool]], str]:
        block_ready, reason = self.block_ready()

        if block_ready:
            block = []
            packet_arrived = []
            for _ in range(self.block_size):
                if self.packet_arrived[self.tail]:
                    block.append(self.data[self.tail])
                    packet_arrived.append(True)
                else:
                    # in practice this would be a memset to zero for this memory
                    block.append(None)
                    packet_arrived.append(False)
                # if we set all packet_arrived to zero then we will
                # know if new data has been written in this section.
                # This is the only time we know for sure that there is
                # no future data being written in this section
                self.packet_arrived[self.tail] = 0
                self.tail = (self.tail + 1) % self.size
            self.current_seq += self.block_size
            self.head = (self.head + self.block_size) % self.size
            return block, packet_arrived, reason
        else:
            return None, None, ""


async def producer(buffers: list[RingBuffer], channels: int) -> None:
    seqs = [0 for _ in range(channels)]
    first_packet = True
    global NUM_DROPPED_PACKETS
    while True:
        c = random.randint(0, channels - 1)
        pkt = Packet(c, seqs[c])

        # keep track of where we're up to for each channel.
        seqs[c] += 1

        # Have some stochastic chance of dropping a packet
        if random.random() >= DROP_PROB:
            buffers[c].add(pkt)

            # Set the initial packet seq for all ring buffers to be the
            # same as the first packet that arrives across any channel.
            # This is simple but may drop a few initial packets
            if first_packet:
                first_packet = False
                for c in range(channels):
                    buffers[c].current_seq = pkt.seq
        else:
            print(f"Dropping packet {pkt.seq} for channel {c}.")
            NUM_DROPPED_PACKETS += 1

        await asyncio.sleep(0.01)


async def consumer(buffers: list[list[RingBuffer]], channels: int) -> None:
    while True:
        for c in range(channels):
            block, packet_arrived, reason = buffers[c].get_block()
            if block:
                # Send off to correlation / beamforming kernels
                # packet_arrived will also get sent off.
                print(
                    f"Sending block for channel {c}: {block}. Packets: {packet_arrived}"
                )
        await asyncio.sleep(0.05)


async def main(
    channels: int = 3,
    buffer_size: int = BUFFER_SIZE,
    block_size: int = BLOCK_SIZE,
):
    buffers = [RingBuffer(buffer_size, block_size, c) for c in range(channels)]
    await asyncio.gather(
        producer(buffers, channels),
        consumer(buffers, channels),
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except:
        print("Exiting...")
        print(f"Number of dropped packets: {NUM_DROPPED_PACKETS}")
        raise
