
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

## Caveats
Exact stats are kept per input file, which determines the maximum memory used. If you find memory problems, break down your input files into smaller ones. A useful idiom to do this with `jq` is
```
zcat segments-9000000.json.gz | jq 'with_entries(select(.key | tonumber <= 9000100))' | gzip > segs100.json.gz
```

When feeding multiple files into `merklizator`, overall stats are carried over as an approximation, using the [t-digest](https://github.com/kpdemetriou/tdigest-cffi) algorithm. This allows to keep running stats for an arbitrary number of blocks and files without ever-growing memory consumption.

If you plan to customize the data gathered and exported by TurboGeth's opcodeTracer, it might be worth it to create and export RoaringBitmaps from TurboGeth instead of the JSON representation of segments / basic blocks. This would avoid the JSON bottleneck in both sides of the pipeline.
