import argparse
import gzip
import json
import logging
import math
import os
import statistics
import time
from typing import Dict, List, TypedDict

from roaringbitmap import RoaringBitmap
from sparklines import sparklines
from tdigest import RawTDigest


def represent_contract(bytemap, chunkmap):
    contract_representation = ""
    for b in range(0, codesize):
        if b % args.chunk_size == 0:
            contract_representation += "|"
        b_in_chunkmap = (b // args.chunk_size) in chunkmap
        char = "."      # this is uninteresting: contract byte that wasn't executed nor chunked
        if b in bytemap:
            char = 'X'  # This is bad: executed bytecode that didn't get merklized
        if b_in_chunkmap:
            char = 'm'  # this is overhead: merklized but unexecuted code
        if b in bytemap and b_in_chunkmap:
            char = "M"  # this is OK: executed and merklized code
        contract_representation += char
    print(contract_representation,"\n")

def clamp(n : int, min_n : int, max_n : int) -> int:
    if n >= max_n:
        return max_n
    if n <= min_n:
        return min_n
    return n

def sparkline_sizes(sizes : List) -> str :
    # sizes is sorted so we can take advantage of that to accelerate things
    median = statistics.median_low(sizes)
    bucket_size = 1
    top_bucket = 2 * median # up to median there's half of the items. Since that's typically a short range anyway, let's show twice that.
    buckets_maxcontent = range(1, top_bucket, bucket_size)  # each bucket contains values up to this, inclusive
    if len(buckets_maxcontent)==0:
        logger.info(f"Can't bucketize, moving on. sizes={sizes}, median={median}, block={block}")
        return f"CAN'T BUCKETIZE! sizes={sizes}"
    buckets_contents = [0 for b in buckets_maxcontent]
    maxbucket = len(buckets_maxcontent)
    count = 0
    for s in sizes:
        b = math.ceil(s / bucket_size)
        if b >= maxbucket:
            # since the sizes are sorted, at this point we know we're finished
            break
        count += 1
        b = clamp(b, 0, len(buckets_contents) - 1)
        buckets_contents[b] += 1

    sl = sparklines(buckets_contents)[0]
    remaining = (1 - count/len(sizes)) * 100
    line = f"median={median}\t\t{buckets_maxcontent[0]}{sl}{buckets_maxcontent[-1]} (+{remaining:.0f}% more)"
    return line


parser = argparse.ArgumentParser(
    description='Reads a directory or a list of json files containing segments from transactions, and applies a chunking strategy to them to calculate the resulting witness sizes',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("traces_dir", help="Directory with trace files in .json.gz format", nargs='+')
parser.add_argument("-s", "--chunk_size", help="Chunk size in bytes", type=int, default=32)
parser.add_argument("-m", "--hash_size", help="Hash size in bytes for construction of the Merkle tree", type=int, default=32)
parser.add_argument("-a", "--arity", help="Number of children per node of the Merkle tree", type=int, default=2)
parser.add_argument("-v", "--log", help="Log level", type=str, default="INFO")
parser.add_argument("-j", "--job_ID", help="ID to distinguish in parallel runs", type=int, default=None)
parser.add_argument("-d", "--detail_level", help="3=transaction, 2=contract, 1=block, 0=file. One level implies the lower ones.", type=int, default=1)

args = parser.parse_args()

loglevel_num = getattr(logging, args.log.upper(), None)
if not isinstance(loglevel_num, int):
    raise ValueError('Invalid log level: %s' % args.loglevel)
logging.basicConfig(level=loglevel_num, format= \
    ("" if args.job_ID is None else f'{args.job_ID:4} | ') + '%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("")

MAXCODESIZE = 0x6000 # from EIP 170

contract_data = TypedDict('contract_data', {'instances':int, 'size':int, 'map':RoaringBitmap}, total=True)

max_chunks = MAXCODESIZE // args.chunk_size
# bitmaps representing the bytes covered by each chunk. They are ANDed with the bitmaps of executed bytes
#chunkificators = MultiRoaringBitmap([RoaringBitmap(range(x * args.chunk_size, (x+1) * args.chunk_size)) for x in range(0, max_chunks + 1)])

# bitmaps representing which nodes in level N of the tree connect to the same parent in level N-1
#merklizators = MultiRoaringBitmap([RoaringBitmap(range(x * args.arity, (x+1) * args.arity)) for x in range(0, len(chunkificators) // args.arity + 1)])

# calculate the number of hashes needed to merklize the given bitmap of chunks
def merklize(chunkmap : RoaringBitmap, arity : int, max_chunks : int):
    # max number of chunks = MAXCODESIZE // chunksize
    # hashes at tree level 0 = as many as chunks
    # max hashes at tree level N = (num of level N-1) / arity
    # we assume fixed number of levels = log (max chunks) / log(arity)
    # L0    0,1,2,3,4,5
    # L1    0   1   2
    # L2    0       1
    # L3    0
    num_levels = math.log(max_chunks) / math.log(arity)
    assert(len(chunkmap) <= max_chunks)
    current_level = 0
    num_hashes = 0
    potential_hashes_in_level = max_chunks
    map = chunkmap
    while potential_hashes_in_level >= arity:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"L{current_level} pot_hashes={potential_hashes_in_level} num_hashes={len(map)}")
        bits = []
        max_hash_in_level = map.max() // arity
        hashes_missing = 0
        for i in range(0, max_hash_in_level + 1):
            siblings_start = i * arity
            siblings_end = siblings_start + arity
            #if not map.isdisjoint(merklizators[i]):
            overlap = map.clamp(siblings_start, siblings_end)  # stop excluded from interval
            lo = len(overlap)
            if lo == 0:
                # these siblings' parent is not present in the tree
                continue
            bits.append(i)
            if lo == arity:
                # we have all the data to recreate the hashes for this node, so we don't need to provide any hash
                continue
            hashes_missing = arity - lo
            num_hashes += hashes_missing

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"L{current_level}: bitmap={bits}, num_missing_hashes={hashes_missing}")

        # switch map to parents'
        map = RoaringBitmap(bits)
        potential_hashes_in_level = math.ceil(potential_hashes_in_level / arity)
        current_level += 1
    return num_hashes

segment_sizes : List[int] = []

if len(args.traces_dir) == 1 and os.path.isdir(args.traces_dir[0]):
    files = sorted([os.path.join(args.traces_dir[0], f) for f in os.listdir(args.traces_dir[0]) if (".json.gz" == f[-8:])])
else:
    files = sorted(args.traces_dir)
total_executed_bytes = 0
total_chunks = 0
total_hashes = 0
total_segsize_digest = RawTDigest()
total_blocks = 0
block : int
print(f"Chunking for tree arity={args.arity}, chunk size={args.chunk_size}, hash size={args.hash_size}")
for f in files:
    t0 = time.time()
    blocks: int = 0
    with gzip.open(f, 'rb') as gzf:
        block_traces = json.load(gzf)
    file_executed_bytes = 0
    file_chunks = 0
    file_hashes = 0
    file_segsizes : List[int] = []
    file_segsize_digest = RawTDigest()
    for block in block_traces:
        traces = block_traces[block]
        blocks += 1
        total_blocks +=1
        if len(traces)==0:
            logger.debug(f"Block {block} is empty")
            continue
        dict_contracts : Dict[str, contract_data] = {}
        reused_contracts = 0
        #block_num_bytes_code=0
        #block_num_bytes_chunks=0
        num_bytes_code = 0
        num_bytes_chunks = 0
        block_segsizes : List[int] = []
        for t in traces:
            tx_hash: str = t["Tx"]
            if tx_hash is not None:
                logger.debug(f"Tx {t['TxAddr']} has {len(t['Segments'])} segments")
                codehash : str = t["CodeHash"]
                data = dict_contracts.get(codehash)
                if data is None:
                    bytemap = RoaringBitmap()
                    instances = 1
                    size = t['CodeSize']
                else:
                    reused_contracts += 1
                    bytemap = data['map']
                    instances = data['instances']+1
                    size = data['size']

                tx_segsizes: List[int] = []
                for s in t["Segments"]:
                    start = s["Start"]
                    end = s["End"]
                    length = end - start + 1
                    range_code = range(start, end+1)
                    bytemap.update(range_code)
                    #bisect.insort_left(segment_sizes, length)
                    tx_segsizes.append(length)
                dict_contracts[codehash] = contract_data(instances=instances, size=size, map=bytemap)
                #del t["Segments"]

                # transaction-level segment stats
                if args.detail_level >= 3:
                    print(f"Block {block} codehash={codehash} tx={t['TxAddr']} segs={len(tx_segsizes)} :"
                          f"segment sizes:{sparkline_sizes(tx_segsizes)}")
                block_segsizes += sorted(tx_segsizes)

        if len(block_segsizes) == 0:
            logger.debug(f"Block {block} had no segments")
            continue

        block_segsizes = sorted(block_segsizes)
        # block-level segment stats
        if args.detail_level >=1:
            stats=sparkline_sizes(block_segsizes)
            print(f"Block {block}: segs={len(block_segsizes):<6d}"+stats)

        block_executed_bytes = 0
        block_hashes = 0
        block_chunks = 0
        block_contract_bytes = 0

        # chunkification of the contracts executed in the block
        for codehash, data in dict_contracts.items():
            instances = data['instances']
            codesize = data['size']
            bytemap = data['map']
            executed_bytes = len(bytemap)
            max_possible_chunk = bytemap.max() // args.chunk_size
            chunksb = RoaringBitmap()
            if args.chunk_size == 1:
                # special case for speed
                chunkmap = bytemap
            else:
                for c in range(0, max_possible_chunk+1):
                    chunk_start = c * args.chunk_size
                    chunk_stop = (c+1) * args.chunk_size
                    #chunkrange = range(chunk_start, chunk_stop)
                    #z1 = len(bytemap.clamp(chunk_start, chunk_stop)) == 0 #fastest
                    #z2 = bytemap.intersection_len(chunkrange) == 0
                    #z3 = bytemap.isdisjoint(chunkrange)
                    #assert(z1 == z2 and z2 == z3)
                    #chunkificator = chunkificators[c]
                    #if not bytemap.isdisjoint(chunkificator):
                    overlap = bytemap.clamp(chunk_start, chunk_stop)
                    if len(overlap) != 0:
                        chunksb.add(c)
                chunkmap = chunksb
            chunks = len(chunkmap)
            chunked_bytes =  chunks * args.chunk_size
            chunked_executed_ratio = chunked_bytes // executed_bytes
            merklization_hashes = merklize(chunkmap, args.arity, max_chunks)
            highlighter : str = ""
            if chunked_executed_ratio > 1:
                highlighter = "\t\t" + "!"*(chunked_executed_ratio-1)
                #represent_contract(bytemap, chunkmap)
            if chunked_bytes < executed_bytes: # sanity check
                logger.info(f"Contract {codehash} in block {block} executes {executed_bytes} but merklizes to {chunked_bytes}")
                highlighter = "\t\t" + "??????"
                represent_contract(bytemap, chunkmap)

            chunk_waste = (1 - executed_bytes / chunked_bytes) * 100
            if args.detail_level >=2:
                print(f"Contract {codehash}: "
                      f"{instances} txs "
                      f"size {codesize}\t"
                      f"executed {executed_bytes}\t"
                      f"chunked {chunked_bytes}\t"
                      f"wasted={chunk_waste:.0f}%\t"
                      f"hashes={merklization_hashes}"
                      +highlighter
                )

            block_executed_bytes += executed_bytes
            block_hashes += merklization_hashes
            block_chunks += chunks

        block_chunk_waste = block_chunks * args.chunk_size - block_executed_bytes
        block_chunk_wasted_ratio = block_chunk_waste / block_chunks * 100
        block_chunk_bytes = block_chunks * args.chunk_size
        block_hash_bytes = block_hashes * args.hash_size
        block_merklization_bytes = block_chunk_bytes + block_hash_bytes
        block_merklization_overhead_ratio = (block_merklization_bytes / block_executed_bytes - 1) * 100

        # block-level merklization stats
        if args.detail_level >=1:
            #logger.info(f"Block {block}: txs={len(traces)}\treused_contracts={reused_contracts}\tcontracts={block_contract_bytes//1024}K\texecuted={block_executed_bytes//1024}K\tchunked={block_chunk_bytes//1024}K\twasted={block_chunk_wasted_ratio:.0f}%")
            print(f"Block {block}: "
                  f"overhead={block_merklization_overhead_ratio:.1f}%\t"
                  f"exec={block_executed_bytes/1024:.1f}K\t"
                  f"merklization= {block_merklization_bytes/1024:.1f} K "
                  f"= {block_chunk_bytes /1024:.1f} K ({block_chunks} chunks) + {block_hash_bytes /1024:.1f} K ({block_hashes} hashes)"
            )

        file_chunks += block_chunks
        file_executed_bytes += block_executed_bytes
        file_hashes += block_hashes
        file_segsizes += block_segsizes
        #file_segsize_digest.batch_update(block_segsizes)
        #file_segsize_digest.compress()

        #total_chunk_bytes += block_chunk_bytes
        #total_executed_bytes += block_executed_bytes
        #total_naive_bytes += block_contract_bytes
        #total_hash_bytes += block_hash_bytes
        #total_segsizes += block_segsizes


    # sorting helps other steps be faster (e.g., stats.median)
    file_segsizes = sorted(file_segsizes)
    for s in file_segsizes:
        total_segsize_digest.insert(s)
    file_chunk_bytes = file_chunks * args.chunk_size
    file_hash_bytes = file_hashes * args.hash_size
    file_merklization_bytes = file_chunk_bytes + file_hash_bytes
    file_merklization_overhead_ratio = (file_merklization_bytes / file_executed_bytes - 1) * 100
    t_file = time.time() - t0
    # file-level merklization stats
    print(
        f"file {f}: "
        f"overhead={file_merklization_overhead_ratio:.1f}%\t"
        f"exec={file_executed_bytes / 1024:.1f}K\t"
        f"merklization= {file_merklization_bytes/1024:.1f} K "
        f"= {file_chunk_bytes / 1024:.1f} K ({file_chunks} chunks) + {file_hash_bytes / 1024:.1f} K ({file_hashes} hashes)\t"
        f"segment sizes:{sparkline_sizes(file_segsizes)}"
    )
    logger.info(f"file {f}: "
                 f"{blocks} blocks in {t_file:.0f} seconds = {blocks/t_file:.1f}bps.")

    total_chunks += file_chunks
    total_executed_bytes += file_executed_bytes
    total_hashes += file_hashes
    total_chunk_bytes = total_chunks * args.chunk_size
    total_hash_bytes = total_hashes * args.hash_size
    total_merklization_bytes = total_chunk_bytes + total_hash_bytes
    total_merklization_overhead_ratio = (total_merklization_bytes / total_executed_bytes - 1) * 100
    print(
        f"running total: "
        f"blocks={total_blocks}\t"
        f"overhead={total_merklization_overhead_ratio:.1f}%\t"
        f"exec={total_executed_bytes / 1024:.1f}K\t"
        f"merklization={total_merklization_bytes/1024:.1f}K = {total_chunk_bytes / 1024:.1f} K ({total_chunks} chunks) + {total_hash_bytes / 1024:.1f} K ({total_hashes} hashes)\t\t"
        f"estimated median:{total_segsize_digest.quantile(0.5):.1f}")
