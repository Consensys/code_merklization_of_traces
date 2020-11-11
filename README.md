
# Code Merklization of transaction traces  
  
`merklificator` is a tool to calculate code merklization outcomes from real world traces, using fixed-size chunking with configurable chunk size, hash size and tree arity.  
  
  ## Install
  Install [pipenv](https://pipenv.pypa.io/en/latest/) in your system, which will allow you to install dependencies for this project with minimum fuss.
 
 Then, cd to this project's directory and run `pipenv shell`. This will open a shell with all the prerequisites installed.
 
In this shell you can run `merklificator`.

## Run
`merklificator` expects traces generated with the opcodeTracer that was merged into TurboGeth at commit [ed1819e](https://github.com/ledgerwatch/turbo-geth/commit/ed1819ec58c0e07e6276406f6c498d650cd3be15). 

Those traces are big (~300MB for a 1000 block file), so you will probably want to gzip them - that's how `merklificator` expects them.

You can find some such trace files in the Releases section.

Then, just run `python merklificator directory_with_traces`.

You can find some sample trace files in the releases section of this repo.

```
$ python merklificator.py --help
usage: merklificator.py [-h] [-s [CHUNK_SIZE [CHUNK_SIZE ...]]] [-m [HASH_SIZE [HASH_SIZE ...]]] [-a ARITY] [-l LOG] [-j JOB_ID] [-d DETAIL_LEVEL] [-g] traces_dir [traces_dir ...]

Reads a directory or multiple files containing segments from transactions, and applies fixed chunking to them to calculate the resulting witness sizes

positional arguments:
  traces_dir            Directory with, or multiple space-separated trace files in .json.gz format

optional arguments:
  -h, --help            show this help message and exit
  -s [CHUNK_SIZE [CHUNK_SIZE ...]], --chunk_size [CHUNK_SIZE [CHUNK_SIZE ...]]
                        Chunk size in bytes. Can be multiple space-separated values. (default: [32])
  -m [HASH_SIZE [HASH_SIZE ...]], --hash_size [HASH_SIZE [HASH_SIZE ...]]
                        Hash size in bytes for construction of the Merkle tree. Can be multiple space-separated values. (default: [32])
  -a ARITY, --arity ARITY
                        Number of children per node of the Merkle tree (default: 2)
  -l LOG, --log LOG     Log level (default: INFO)
  -j JOB_ID, --job_ID JOB_ID
                        ID to distinguish in parallel runs (default: None)
  -d DETAIL_LEVEL, --detail_level DETAIL_LEVEL
                        3=transaction, 2=contract, 1=block, 0=file. One level implies the lower ones. (default: 1)
  -g, --segment_stats   Whether to calculate and show segments stats (default: False)

NOTE: hash_sizes have no effect on the tree calculations; they are just just for convenience to calculate overheads.
```

## Results
This is an example of the output of `merklificator` for 40/32/24 byte chunks, 32/24/16/8 byte hashes, 2-arity (binary) tries and with segment details:
```
...
Block 9000100: exec=96.4K	segs=9110	seg_sizes:median=17		1▁▁▂▂▂▃▆▄█▆▄▇▆▄▄▄▂▄▄▂▄▃▄▅▂▃▂▃▁▄▂▁▁33 (+19% more)
	chunksize=40	chunk_oh= 36.6% (3.37k chunks) + 2.59k hashes	
		hashsize=32	 hash_oh= 84.0%		total_oh=120.6%
		hashsize=24	 hash_oh= 63.0%		total_oh= 99.6%
		hashsize=16	 hash_oh= 42.0%		total_oh= 78.6%
		hashsize= 8	 hash_oh= 21.0%		total_oh= 57.6%
	chunksize=32	chunk_oh= 29.6% (4k chunks) + 2.84k hashes	
		hashsize=32	 hash_oh= 91.9%		total_oh=121.5%
		hashsize=24	 hash_oh= 68.9%		total_oh= 98.5%
		hashsize=16	 hash_oh= 46.0%		total_oh= 75.5%
		hashsize= 8	 hash_oh= 23.0%		total_oh= 52.5%
	chunksize=24	chunk_oh= 23.0% (5.06k chunks) + 3.13k hashes	
		hashsize=32	 hash_oh=101.4%		total_oh=124.4%
		hashsize=24	 hash_oh= 76.1%		total_oh= 99.0%
		hashsize=16	 hash_oh= 50.7%		total_oh= 73.7%
		hashsize= 8	 hash_oh= 25.4%		total_oh= 48.3%
file segs.json.gz: exec=9092.2K	segs=1470396	seg_sizes:median=14		1▁▁▁▃▂▃▇▆█▇▅▇▇▅▅▃▃▄▄▄▃▃▃▄▃▃▂27 (+22% more)
	chunksize=40	chunk_oh= 34.4% (312.77k chunks) + 218.92k hashes	
		hashsize=32	 hash_oh= 75.2%		total_oh=109.6%
		hashsize=24	 hash_oh= 56.4%		total_oh= 90.8%
		hashsize=16	 hash_oh= 37.6%		total_oh= 72.0%
		hashsize= 8	 hash_oh= 18.8%		total_oh= 53.2%
	chunksize=32	chunk_oh= 28.6% (374.24k chunks) + 238.72k hashes	
		hashsize=32	 hash_oh= 82.0%		total_oh=110.7%
		hashsize=24	 hash_oh= 61.5%		total_oh= 90.2%
		hashsize=16	 hash_oh= 41.0%		total_oh= 69.7%
		hashsize= 8	 hash_oh= 20.5%		total_oh= 49.1%
	chunksize=24	chunk_oh= 22.3% (474.58k chunks) + 264.52k hashes	
		hashsize=32	 hash_oh= 90.9%		total_oh=113.3%
		hashsize=24	 hash_oh= 68.2%		total_oh= 90.5%
		hashsize=16	 hash_oh= 45.5%		total_oh= 67.8%
		hashsize= 8	 hash_oh= 22.7%		total_oh= 45.1%
```
This shows the stats for block 9000100 and for all the blocks in the input file. 
Reporting is similar at the block, file and global level, so we'll focus on the file.

The file contains 101 blocks, which contain 1470396 segments, with a median length of 14 bytes, and the graphed distribution. The histogram shows the range of lengths 1 to 27, and 22% of lenghts are larger than that.  
The distribution is only shown up to 2*median because it typically is very skewed towards shorter lengths.

In the file, the executed bytecode comprised 9092.2KB, and its merklization caused an overhead dependent on the chunk size and the hash size. Importantly, chunk size drives the merklization and decides the number of hashes that are needed, so this is reflected in the report.  

Additionally, for each given hash size, the resulting overhead owed to the hashes is calculated. `chunk_oh` is the chunk overhead, `hash_oh` is the hash overhead, and `total_oh` is the total overhead, with each figure relative to the number of executed bytes. So an overhead of 0% would mean that the only data transmitted was the executed bytes - which is only possible in the case of sending the whole contract bytecode, since that would moot the need for chunking and hashing.  

In the case of the present file, the minimum total overhead in the range of given parameters happens with chunksize=24 and hashsize=8. If we restricted the hashsize to 32, then the minimum would be at chunksize=40.

All statistics are exact for the block and file levels. For multiple files, the segment median length is approximated to avoid unbounded memory growth, but the overhead calculations are still exact. 

## Caveats
Exact stats are kept per input file, which determines the maximum memory used. If you find memory problems, break down your input files into smaller ones. A useful idiom to do this with `jq` is
```
zcat segments-9000000.json.gz | jq 'with_entries(select(.key | tonumber <= 9000100))' | gzip > segs100.json.gz
```

When feeding multiple files into `merklizator`, overall stats are carried over as an approximation, using the [t-digest](https://github.com/kpdemetriou/tdigest-cffi) algorithm. This allows to keep running stats for an arbitrary number of blocks and files without ever-growing memory consumption.

If you plan to customize the data gathered and exported by TurboGeth's opcodeTracer, it might be worth it to create and export RoaringBitmaps from TurboGeth instead of the JSON representation of segments / basic blocks. This would avoid the JSON bottleneck in both sides of the pipeline.
