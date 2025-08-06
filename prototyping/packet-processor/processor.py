import asyncio
import random
from typing import Union

NUM_PACKETS_FOR_PROCESSING = 50
NUM_BLOCKS_IN_BUFFER = 5
BUFFER_SIZE = NUM_PACKETS_FOR_PROCESSING * NUM_BLOCKS_IN_BUFFER
BLOCK_SIZE = NUM_PACKETS_FOR_PROCESSING
DROP_PROB = 0.001


class Packet:
    def __init__(self, receiver: int, channel: int, seq: int):
        self.receiver = receiver
        self.channel = channel
        self.seq = seq

    def __repr__(self):
        return f"Pkt(r={self.receiver}, c={self.channel}, seq={self.seq})"


class RingBuffer:
    def __init__(self, size: int, block_size: int, receiver: int, channel: int):
        self.size = size
        self.data = [None] * size
        self.packet_arrived = [0] * size
        self.head = 0  # next write position
        self.tail = 0  # next read position
        self.current_seq = 0  # current block sequence number
        self.last_arrived_seq = 0
        self.block_size = block_size
        self.receiver = receiver
        self.channel = channel

    def add(self, pkt: Packet) -> None:
        if self.current_seq == 0:
            self.current_seq = pkt.seq

        if pkt.seq < self.current_seq:
            print(
                f"Dropping packet with seq num {pkt.seq} as it is behind the current block head {self.current_seq} for receiver {self.receiver} and channel {self.channel}."
            )

        if pkt.seq - self.current_seq > self.size:
            raise ValueError("Buffer has eaten its own tail!")
        next_idx = (self.head + (pkt.seq - self.current_seq)) % self.size
        self.data[next_idx] = pkt
        self.packet_arrived[next_idx] = 1
        self.last_arrived_seq = pkt.seq

    def block_ready(self) -> bool:
        # If we assume no out-of-order blocks, then arrival of later block means
        # that no more are coming for this block
        return self.last_arrived_seq >= self.current_seq + self.block_size

    def get_block(self) -> Union[None, list[Packet]]:
        if self.block_ready():
            block = []
            for _ in range(self.block_size):
                if self.packet_arrived[self.tail]:
                    block.append(self.data[self.tail])
                else:
                    # in practice this would be a memset to zero for this memory
                    block.append(None)
                # if we set all packet_arrived to zero then we will
                # know if new data has been written in this section.
                # This is the only time we know for sure that there is
                # no future data being written in this section
                self.packet_arrived[self.tail] = 0
                self.tail = (self.tail + 1) % self.size
            self.current_seq += self.block_size
            self.head = (self.head + self.block_size) % self.size
            return block
        else:
            return None


async def producer(
    buffers: list[list[RingBuffer]], receivers: int, channels: int
) -> None:
    seqs = [[0 for _ in range(channels)] for _ in range(receivers)]
    first_packet = True
    while True:
        r = random.randint(0, receivers - 1)
        c = random.randint(0, channels - 1)
        pkt = Packet(r, c, seqs[r][c])
        seqs[r][c] += 1

        if random.random() >= DROP_PROB:
            buffers[r][c].add(pkt)

            # Set the initial packet seq for all ring buffers to be the
            # same as the first packet that arrives across any channel.
            # This is simple but may drop a few initial packets
            if first_packet:
                first_packet = False
                for r in range(receivers):
                    for c in range(channels):
                        buffers[r][c].current_seq = pkt.seq
        else:
            print(f"Dropping packet {pkt.seq} for receiver {r} and channel {c}.")

        await asyncio.sleep(0.01)


async def consumer(
    buffers: list[list[RingBuffer]], receivers: int, channels: int
) -> None:
    while True:
        for r in range(receivers):
            for c in range(channels):
                block = buffers[r][c].get_block()
                if block:
                    print(f"Sending block for receiver {r}, channel {c}:", block)
        await asyncio.sleep(0.05)


async def main(
    receivers: int = 4,
    channels: int = 3,
    buffer_size: int = BUFFER_SIZE,
    block_size: int = BLOCK_SIZE,
):
    buffers = [
        [RingBuffer(buffer_size, block_size, r, c) for c in range(channels)]
        for r in range(receivers)
    ]
    await asyncio.gather(
        producer(buffers, receivers, channels),
        consumer(buffers, receivers, channels),
    )


if __name__ == "__main__":
    asyncio.run(main())
