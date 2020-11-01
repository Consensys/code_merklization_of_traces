import collections
import json
import logging
import math
import argparse
import os
import gzip
import statistics
import time

from roaringbitmap import RoaringBitmap, ImmutableRoaringBitmap, MultiRoaringBitmap
from typing import Dict, List, TypedDict, AnyStr
from collections import defaultdict
from sparklines import sparklines

def represent_contract(bytemap, chunkmap):
    contract_representation = ""
    for b in range(0, codesize):
        if b % args.chunk_size == 0:
            contract_representation += "|"
        b_in_chunkmap = (b // args.chunk_size) in chunkmap
        char = "."
        if b in bytemap:
            char = 'X'
        if b_in_chunkmap:
            char = 'm'
        if b in bytemap and b_in_chunkmap:
            char = "O"
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
    top_bucket = 2*median # up to median there's half of the items. Since that's typically a short range anyway, let's show some more.
    buckets_max = range(2, top_bucket, bucket_size)  # each bucket contains values up to this, inclusive
    buckets_contents = [0 for b in buckets_max]
    count = 0
    for s in sizes:
        i = math.ceil(s / bucket_size)
        if i >= len(buckets_contents):
            break
        count += 1
        i = clamp(i, 0, len(buckets_contents) - 1)
        buckets_contents[i] += 1

    sl = sparklines(buckets_contents)[0]
    remaining = (1 - count/len(sizes)) * 100
    # stats=f"\t\tavg={mean:.0f}\t\tstdev={sd:.0f}\t\tmedian={median}\t\t2{sl}{buckets_max[-1]}+"
    line = f"\t\tmedian={median}\t\t{buckets_max[0]}{sl}{buckets_max[-1]} (+{remaining:.0f}% more)"
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
chunkificators = MultiRoaringBitmap([RoaringBitmap(range(x * args.chunk_size, (x+1) * args.chunk_size)).freeze() for x in
                  range(0, max_chunks + 1)])

# bitmaps representing which nodes in level N of the tree connect to the same parent in level N-1
merklizators = MultiRoaringBitmap([RoaringBitmap(range(x * args.arity, (x+1) * args.arity)).freeze() for x in
                range(0, len(chunkificators) // args.arity + 1)])

# calculate the number of hashes needed to merklize the given bitmap of chunks
def merklize(chunkmap : ImmutableRoaringBitmap):
    # max number of chunks = MAXCODESIZE // chunksize
    # hashes at tree level 0 = as many as chunks
    # max hashes at tree level N = (num of level N-1) / arity
    # we assume fixed number of levels = log (max chunks) / log(arity)
    # L0    0,1,2,3,4,5
    # L1    0   1   2
    # L2    0       1
    # L3    0
    num_levels = math.log(max_chunks) / math.log(args.arity)
    levels_done = 0
    num_hashes = 0
    potential_hashes_in_level = max_chunks
    map = chunkmap
    while potential_hashes_in_level > args.arity:
        logging.debug(f"level {levels_done} pot_hashes={potential_hashes_in_level} num_hashes={len(map)}")
        if levels_done > 0:
            num_hashes += len(map)
        bits = []
        max_hash_in_level = map.max() // args.arity
        for i in range(0, max_hash_in_level + 1):
            if not map.isdisjoint(merklizators[i]):
                bits.append(i)
        # switch map to parents'
        map = RoaringBitmap(bits)
        potential_hashes_in_level = math.ceil(potential_hashes_in_level / args.arity)
        levels_done += 1
    num_hashes += 1 # the root
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

t0 = time.time()
block : int
blocks : int = 0

print(f"Chunking for tree arity={args.arity}, chunk size={args.chunk_size}, hash size={args.hash_size}")
for f in files:
    with gzip.open(f, 'rb') as gzf:
        block_traces = json.load(gzf)
        file_executed_bytes = 0
        file_chunk_bytes = 0
        file_contract_bytes = 0
        file_hash_bytes = 0
        file_segsizes : List[int] = []
        for block in block_traces:
            traces = block_traces[block]
            blocks += 1
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
                        bytemap = RoaringBitmap().clamp(0,MAXCODESIZE)
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
                        sd = 0
                        if len(tx_segsizes)>2:
                            sd = statistics.stdev(tx_segsizes)
                            qt = statistics.quantiles(tx_segsizes)
                        print(f"Block {block} codehash={codehash} tx={t['TxAddr']} segs={len(tx_segsizes)} :"
                                     f"\t\tavg={statistics.mean(tx_segsizes):.0f}\t\tstdev={sd:.0f}\t\tntiles={qt}")
                    block_segsizes += sorted(tx_segsizes)

            if len(block_segsizes) == 0:
                logging.debug(f"Block {block} had no segments")
                continue

            block_segsizes = sorted(block_segsizes)
            # block-level segment stats
            if args.detail_level >=1:
                stats=sparkline_sizes(block_segsizes)
                print(f"Block {block}: segs={len(block_segsizes)}"+stats)

            block_executed_bytes = 0
            block_chunk_bytes = 0
            block_contract_bytes = 0

            # chunkification
            for codehash, data in dict_contracts.items():
                instances = data['instances']
                codesize = data['size']
                bytemap = data['map'].freeze()
                executed_bytes = len(bytemap)
                max_possible_chunk = bytemap.max() // args.chunk_size
                chunks = []
                for c in range(0, max_possible_chunk+1):
                    if not bytemap.isdisjoint(chunkificators[c]):
                        chunks.append(c)
                chunkmap = RoaringBitmap(chunks).clamp(0,len(chunkificators))
                chunked_bytes = len(chunks) * args.chunk_size
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
            block_hash_bytes = merklize(chunkmap) * args.hash_size
            block_merklization_bytes = block_chunk_bytes + block_hash_bytes
            block_merklization_overhead_ratio = (block_merklization_bytes / block_executed_bytes - 1) * 100

            # block-level merklization stats
            if args.detail_level >=1:
                #logging.info(f"Block {block}: txs={len(traces)}\treused_contracts={reused_contracts}\tcontracts={block_contract_bytes//1024}K\texecuted={block_executed_bytes//1024}K\tchunked={block_chunk_bytes//1024}K\twasted={block_chunk_wasted_ratio:.0f}%")
                print(f"Block {block}: overhead={block_merklization_overhead_ratio:.1f}%\texec={block_executed_bytes/1024:.0f}K\tmerklization= {block_merklization_bytes/1024:.1f} K = {block_chunk_bytes/1024:.1f} + {block_hash_bytes/1024:.1f} K")

            file_chunk_bytes += block_chunk_bytes
            file_executed_bytes += block_executed_bytes
            file_contract_bytes += block_contract_bytes
            file_hash_bytes += block_hash_bytes
            file_segsizes += block_segsizes

            #total_chunk_bytes += block_chunk_bytes
            #total_executed_bytes += block_executed_bytes
            #total_naive_bytes += block_contract_bytes
            #total_hash_bytes += block_hash_bytes
            #total_segsizes += block_segsizes

    file_segsizes = sorted(file_segsizes)
    file_merklization_bytes = file_chunk_bytes + file_hash_bytes
    file_merklization_overhead_ratio = (file_merklization_bytes / file_executed_bytes - 1) * 100
    t = time.time() - t0
    # file-level merklization stats
    print(
        f"file {f}:\toverhead={file_merklization_overhead_ratio:.1f}%\texec={file_executed_bytes / 1024:.0f}K\t"
        f"merklization={file_merklization_bytes/1024:.1f}K = {file_chunk_bytes / 1024:.1f} K chunks + {file_hash_bytes / 1024:.1f} K hashes\t"
        f"segment sizes:{sparkline_sizes(file_segsizes)}")
    logging.info(f"file {f} done in {t:.0f} seconds: {blocks/t:0.1f}bps")

    total_chunk_bytes += file_chunk_bytes
    total_executed_bytes += file_executed_bytes
    total_naive_bytes += file_contract_bytes
    total_hash_bytes += file_hash_bytes

    total_merklization_bytes = total_chunk_bytes + total_hash_bytes
    total_merklization_overhead_ratio = (total_merklization_bytes / total_executed_bytes - 1) * 100
    print(
        f"running total:\toverhead={total_merklization_overhead_ratio:.1f}%\texec={total_executed_bytes / 1024:.0f}K\t"
        f"merklization={total_merklization_bytes/1024:.1f}K = {total_chunk_bytes / 1024:.1f} K chunks + {total_hash_bytes / 1024:.1f} K hashes\t")
