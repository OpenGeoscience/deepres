# Running the pipeline

### 1. Install dependencies
```
pip install -r reqiurements.txt
```

### 2. Start Luigi Daemon
```
cd preprocess
luigid
```

You can go to [Luigi UI](http://localhost:8082) to check your tasks.
Also you can view your [task history](http://localhost:8082/history).

### 3. Run the ingest pipeline
```
cd preprocess
python tasks.py IngestPipeline --input-directory /path/to/your/input/directory/ --output-directory /path/to/your/desired/output/directory --ground-truth /path/to/the/cdl/file/2017_30m_cdls.img --workers 8
```

#### Parameters
- --input-directory: This is the directory of tiff files. For example the /ark directory is a good candidate.
- --output-directory: This is where the tiles will be stored.
- --ground-truth: This is the cdl file in our case.
- --workers: Number of workers for luigi.
