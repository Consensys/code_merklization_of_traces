import json
import argparse
import gzip
import json
import logging
import math
import os
import statistics
import time
from typing import Dict, List, TypedDict

from roaringbitmap import RoaringBitmap, ImmutableRoaringBitmap
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
    bucket_size = 2
    top_bucket = 2 * median # up to median there's half of the items. Since that's typically a short range anyway, let's show twice that.
    buckets_maxcontent = range(2, top_bucket, bucket_size)  # each bucket contains values up to this, inclusive
    if len(buckets_maxcontent)==0:
        logging.info(f"Can't bucketize, moving on. sizes={sizes}, median={median}, block={block}")
        return f"CAN'T BUCKETIZE! sizes={sizes}"
    buckets_contents = [0 for b in buckets_maxcontent]
    maxbucket = len(buckets_maxcontent)
    count = 0
    for s in sizes:
        i = math.ceil(s / bucket_size)
        if i >= maxbucket:
            break
        count += 1
        i = clamp(i, 0, len(buckets_contents) - 1)
        buckets_contents[i] += 1

    sl = sparklines(buckets_contents)[0]
    remaining = (1 - count/len(sizes)) * 100
    line = f"median={median}\t\t{buckets_maxcontent[0]}{sl}{buckets_maxcontent[-1]} (+{remaining:.0f}% more)"
    return line

parser = argparse.ArgumentParser(description='Apply chunking to transaction segments',
                                 epilog='Reads a directory or a list of json files containing segments from transactions, and applies a chunking strategy to them to calculate the resulting witness sizes')
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

MAXCODESIZE = 0x6000 # from EIP 170

contract_data = TypedDict('contract_data', {'instances':int, 'size':int, 'map':RoaringBitmap}, total=True)

max_chunks = MAXCODESIZE // args.chunk_size
# bitmaps representing the bytes covered by each chunk. They are ANDed with the bitmaps of executed bytes
#chunkificators = MultiRoaringBitmap([RoaringBitmap(range(x * args.chunk_size, (x+1) * args.chunk_size)).freeze() for x in range(0, max_chunks + 1)])

# bitmaps representing which nodes in level N of the tree connect to the same parent in level N-1
#merklizators = MultiRoaringBitmap([RoaringBitmap(range(x * args.arity, (x+1) * args.arity)).freeze() for x in range(0, len(chunkificators) // args.arity + 1)])

# calculate the number of hashes needed to merklize the given bitmap of chunks
def merklize(chunkmap : ImmutableRoaringBitmap, arity : int, max_chunks : int):
    # max number of chunks = MAXCODESIZE // chunksize
    # hashes at tree level 0 = as many as chunks
    # max hashes at tree level N = (num of level N-1) / arity
    # we assume fixed number of levels = log (max chunks) / log(arity)
    # L0    0,1,2,3,4,5
    # L1    0   1   2
    # L2    0       1
    # L3    0
    num_levels = math.log(max_chunks) / math.log(arity)
    current_level = 0
    num_hashes = 0
    potential_hashes_in_level = max_chunks
    map = chunkmap
    while potential_hashes_in_level >= arity:
        logging.debug(f"L{current_level} pot_hashes={potential_hashes_in_level} num_hashes={len(map)}")
        bits = []
        max_hash_in_level = map.max() // arity
        hashes_missing = 0
        for i in range(0, max_hash_in_level + 1):
            siblings_start = i * arity
            siblings_end = (i + 1) * arity
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

        logging.debug(f"L{current_level}: bitmap={bits}, num_missing_hashes={hashes_missing}")

        # switch map to parents'
        map = RoaringBitmap(bits).freeze()
        potential_hashes_in_level = math.ceil(potential_hashes_in_level / arity)
        current_level += 1
    return num_hashes

segment_sizes : List[int] = []

if len(args.traces_dir) == 1 and os.path.isdir(args.traces_dir[0]):
    files = sorted([os.path.join(args.traces_dir[0], f) for f in os.listdir(args.traces_dir[0]) if (".json.gz" == f[-8:])])
else:
    files = sorted(args.traces_dir)
total_executed_bytes = 0
total_chunk_bytes = 0
total_naive_bytes = 0
total_hash_bytes = 0
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
    file_chunk_bytes = 0
    file_contract_bytes = 0
    file_hash_bytes = 0
    file_segsizes : List[int] = []
    file_segsize_digest = RawTDigest()
    for block in block_traces:
        traces = block_traces[block]
        blocks += 1
        total_blocks +=1
        if len(traces)==0:
            logging.debug(f"Block {block} is empty")
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
                logging.debug(f"Tx {t['TxAddr']} has {len(t['Segments'])} segments")
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
            logging.debug(f"Block {block} had no segments")
            continue

        block_segsizes = sorted(block_segsizes)
        # block-level segment stats
        if args.detail_level >=1:
            stats=sparkline_sizes(block_segsizes)
            print(f"Block {block}: segs={len(block_segsizes):<6d}"+stats)

        block_executed_bytes = 0
        block_chunk_bytes = 0
        block_contract_bytes = 0

        # chunkification of the contracts executed in the block
        for codehash, data in dict_contracts.items():
            instances = data['instances']
            codesize = data['size']
            bytemap : ImmutableRoaringBitmap = data['map'].freeze()
            executed_bytes = len(bytemap)
            max_possible_chunk = bytemap.max() // args.chunk_size
            chunks = []
            if args.chunk_size == 1:
                # special case for speed
                chunkmap = bytemap
            else:
                for c in range(0, max_possible_chunk+1):
                    chunk_start = c * args.chunk_size
                    chunk_stop = (c+1) * args.chunk_size
                    #chunkrange = range(chunkstart, chunkstop)
                    #z1 = len(bytemap.clamp(chunkstart, chunkstop)) == 0
                    #z2 = bytemap.intersection_len(chunkrange) == 0
                    #z3 = bytemap.isdisjoint(chunkrange)
                    #assert(z1 == z2 and z2 == z3)
                    #chunkificator = chunkificators[c]
                    #if not bytemap.isdisjoint(chunkificator):
                    overlap = bytemap.clamp(chunk_start, chunk_stop)
                    if len(overlap) != 0:
                        chunks.append(c)
                chunkmap = RoaringBitmap(chunks).freeze()
            chunked_bytes = len(chunkmap) * args.chunk_size
            chunked_executed_ratio = chunked_bytes // executed_bytes
            highlighter : str = ""
            if chunked_executed_ratio > 1:
                highlighter = "\t\t" + "!"*(chunked_executed_ratio-1)
                #represent_contract(bytemap, chunkmap)
            if chunked_bytes < executed_bytes: # sanity check
                logging.info(f"Contract {codehash} in block {block} executes {executed_bytes} but merklizes to {chunked_bytes}")
                highlighter = "\t\t" + "??????"
                represent_contract(bytemap, chunkmap)

            chunk_waste = (1 - executed_bytes / chunked_bytes) * 100
            if args.detail_level >=2:
                print(f"Contract {codehash}: {instances} txs, size {codesize}\texecuted {executed_bytes}\tchunked {chunked_bytes}\twasted={chunk_waste:.0f}%"+highlighter)

            block_chunk_bytes += chunked_bytes
            block_contract_bytes += codesize
            block_executed_bytes += executed_bytes

        block_chunk_waste = block_chunk_bytes - block_executed_bytes
        block_chunk_wasted_ratio = block_chunk_waste / block_chunk_bytes * 100
        block_hash_bytes = merklize(chunkmap, args.arity, max_chunks) * args.hash_size
        block_merklization_bytes = block_chunk_bytes + block_hash_bytes
        block_merklization_overhead_ratio = (block_merklization_bytes / block_executed_bytes - 1) * 100

        # block-level merklization stats
        if args.detail_level >=1:
            #logging.info(f"Block {block}: txs={len(traces)}\treused_contracts={reused_contracts}\tcontracts={block_contract_bytes//1024}K\texecuted={block_executed_bytes//1024}K\tchunked={block_chunk_bytes//1024}K\twasted={block_chunk_wasted_ratio:.0f}%")
            print(f"Block {block}: overhead={block_merklization_overhead_ratio:.1f}%\texec={block_executed_bytes/1024:.1f}K\tmerklization= {block_merklization_bytes/1024:.1f} K = {block_chunk_bytes/1024:.1f} + {block_hash_bytes/1024:.1f} K")

        file_chunk_bytes += block_chunk_bytes
        file_executed_bytes += block_executed_bytes
        file_contract_bytes += block_contract_bytes
        file_hash_bytes += block_hash_bytes
        file_segsizes += block_segsizes
        #file_segsize_digest.batch_update(block_segsizes)
        #file_segsize_digest.compress()

        #total_chunk_bytes += block_chunk_bytes
        #total_executed_bytes += block_executed_bytes
        #total_naive_bytes += block_contract_bytes
        #total_hash_bytes += block_hash_bytes
        #total_segsizes += block_segsizes


    file_segsizes = sorted(file_segsizes)
    for s in file_segsizes:
        total_segsize_digest.insert(s)
    file_merklization_bytes = file_chunk_bytes + file_hash_bytes
    file_merklization_overhead_ratio = (file_merklization_bytes / file_executed_bytes - 1) * 100
    t_file = time.time() - t0
    # file-level merklization stats
    print(
        f"file {f}:\toverhead={file_merklization_overhead_ratio:.1f}%\texec={file_executed_bytes / 1024:.1f}K\t"
        f"merklization={file_merklization_bytes/1024:.1f}K = {file_chunk_bytes / 1024:.1f} K chunks + {file_hash_bytes / 1024:.1f} K hashes\t"
        f"segment sizes:{sparkline_sizes(file_segsizes)}")
    logging.info(f"file {f}: {blocks} blocks in {t_file:.0f} seconds = {blocks/t_file:.1f}bps.")

    total_chunk_bytes += file_chunk_bytes
    total_executed_bytes += file_executed_bytes
    total_naive_bytes += file_contract_bytes
    total_hash_bytes += file_hash_bytes

    total_merklization_bytes = total_chunk_bytes + total_hash_bytes
    total_merklization_overhead_ratio = (total_merklization_bytes / total_executed_bytes - 1) * 100
    print(
        f"running total: blocks={total_blocks}\toverhead={total_merklization_overhead_ratio:.1f}%\texec={total_executed_bytes / 1024:.1f}K\t"
        f"merklization={total_merklization_bytes/1024:.1f}K = {total_chunk_bytes / 1024:.1f} K chunks + {total_hash_bytes / 1024:.1f} K hashes\t"
        f"\testimated median:{total_segsize_digest.quantile(0.5):.1f}")
