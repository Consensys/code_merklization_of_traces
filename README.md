
# Code Merklization of transaction traces  
  
`merklificator` is a tool to calculate code merklization outcomes from real world traces, using fixed-size chunking with configurable chunk size, hash size and tree arity.  
  
  ## Install
  Install [pipenv](https://pipenv.pypa.io/en/latest/) in your system, which will allow you to install dependencies for this project with minimum fuss.
 
 Then, cd to this project's directory and run `pipenv shell`. This will open a shell with all the prerequisites installed.
 
In this shell you can run `merklificator`.

## Run
`merklificator` expects traces generated with the opcodeTracer that was merged into TurboGeth at commit [ed1819e](https://github.com/ledgerwatch/turbo-geth/commit/ed1819ec58c0e07e6276406f6c498d650cd3be15).

Those traces are big (~300MB for a 1000 block file), so you will probably want to gzip them - that's how `merklificator` expects them.

Then, just run `python merklificator directory_with_traces`.

You can find some sample trace files in the releases section of this repo.

```
$ python merklificator.py --help
usage: merklificator.py [-h] [-s CHUNK_SIZE] [-m HASH_SIZE] [-a ARITY] [-v LOG] [-j JOB_ID] [-d DETAIL_LEVEL] traces_dir [traces_dir ...]

Reads a directory or a list of json files containing segments from transactions, and applies a chunking strategy to them to calculate the resulting witness sizes

positional arguments:
  traces_dir            Directory with trace files in .json.gz format

optional arguments:
  -h, --help            show this help message and exit
  -s CHUNK_SIZE, --chunk_size CHUNK_SIZE
                        Chunk size in bytes (default: 32)
  -m HASH_SIZE, --hash_size HASH_SIZE
                        Hash size in bytes for construction of the Merkle tree (default: 32)
  -a ARITY, --arity ARITY
                        Number of children per node of the Merkle tree (default: 2)
  -v LOG, --log LOG     Log level (default: INFO)
  -j JOB_ID, --job_ID JOB_ID
                        ID to distinguish in parallel runs (default: None)
  -d DETAIL_LEVEL, --detail_level DETAIL_LEVEL
                        3=transaction, 2=contract, 1=block, 0=file. One level implies the lower ones. (default: 1)
```

## Results
This is an example of the output of `merklizator` for 32-byte chunks, 32-byte hashes and 2-arity (binary) tries:
```
Block 9000095: segs=14270 median=18		2▁▁▂▄█▆█▅▃▆▄▃▄▃▃▂▂34 (+25% more)
Block 9000095: overhead=110.2%	exec=141.1K	chunks=5743	hashes=3745	merklization= 296.5 K = 179.5 + 117.0 K
Block 9000096: segs=13526 median=18		2▁▂▂▄█▇█▅▃▅▄▄▅▅▃▂▂34 (+21% more)
Block 9000096: overhead=104.5%	exec=109.1K	chunks=4390	hashes=2749	merklization= 223.1 K = 137.2 + 85.9 K
Block 9000097: segs=15607 median=15		2▁▁▂▄▆▆█▅▃▄▄▃▃▃28 (+26% more)
Block 9000097: overhead=111.5%	exec=138.3K	chunks=5659	hashes=3703	merklization= 292.6 K = 176.8 + 115.7 K
Block 9000098: segs=14042 median=16		2▁▁▂▅█▆█▄▃▅▃▃▃▃▃30 (+26% more)
Block 9000098: overhead=107.8%	exec=164.9K	chunks=6764	hashes=4200	merklization= 342.6 K = 211.4 + 131.2 K
Block 9000099: segs=9897  median=14		2▁▂▃▅▇▆█▄▂▃▅▃▃26 (+26% more)
Block 9000099: overhead=116.5%	exec=135.4K	chunks=5609	hashes=3776	merklization= 293.3 K = 175.3 + 118.0 K
Block 9000100: segs=9110  median=17		2▁▁▂▅█▆█▅▃▅▄▅▄▃▂▄32 (+20% more)
Block 9000100: overhead=121.5%	exec=96.4K	chunks=3997	hashes=2836	merklization= 213.5 K = 124.9 + 88.6 K
file /Users/mija/IGNORED BY TIMEMACHINE/traces-test/segs.json.gz: overhead=110.7%	exec=9092.2K	merklization= 19155.2 K = 11695.1 K chunks + 7460.1 K hashes	segment sizes:median=14		2▁▁▂▆█▇█▅▃▄▄▃▄26 (+25% more)
16:16:25 file /Users/mija/IGNORED BY TIMEMACHINE/traces-test/segs.json.gz: 101 blocks in 11 seconds = 9.0bps.
running total: blocks=101	overhead=110.7%	exec=9092.2K	merklization=19155.2K = 11695.1 K chunks + 7460.1 K hashes		estimated median:14.0
```
This means that for block 9000095, the executed bytecode is 141.1KB, and the merklization is 179.5KB in 5743 chunks + 117.0K in 3745 hashes, totaling 296.5KB. Therefore the merklization caused a 104.5% of overhead over just sending the executed bytecode.

In that same block there were 13526 segments (AKA basic blocks), and their median length was 18 bytes, with their length having the distribution shown to the right. The leftmost column of the sparkline shows length=2 and the rightmost column shows length=34. Apart from that, 25% of the samples had lengths over that range, and so didn't appear in the sparkline.

The sparkline only shows the range 2 to (2*median) because the distribution of lengths usually has a long, sparse tail that makes it harder to see the detail of the shorter lengths.

The statistics for these blocks are carried over to get exact stats for the whole trace file. See line starting with `file segs.json.gz`.  

Finally, the stats for all the files analyzed in the present run are accumulated into running stats, which are reported after each file.

## Caveats
Exact stats are kept per input file, which determines the maximum memory used. If you find memory problems, break down your input files into smaller ones. A useful idiom to do this with `jq` is
```
zcat segments-9000000.json.gz | jq 'with_entries(select(.key | tonumber <= 9000100))' | gzip > segs100.json.gz
```

When feeding multiple files into `merklizator`, overall stats are carried over as an approximation, using the [t-digest](https://github.com/kpdemetriou/tdigest-cffi) algorithm. This allows to keep running stats for an arbitrary number of blocks and files without ever-growing memory consumption.

If you plan to customize the data gathered and exported by TurboGeth's opcodeTracer, it might be worth it to create and export RoaringBitmaps from TurboGeth instead of the JSON representation of segments / basic blocks. This would avoid the JSON bottleneck in both sides of the pipeline.
