from tqdm import tqdm
from pathlib import Path
from hashlib import sha256
from pyarrow import dataset as ds
from huggingface_hub import snapshot_download

OUTPUT_DIR = Path("./10BT")
def grab_fineweb():
    DATA_DIR = Path(snapshot_download(
        "HuggingFaceFW/fineweb", repo_type="dataset", allow_patterns="sample/10BT/*"
    )) / 'sample/10BT'
    dataset = ds.dataset(DATA_DIR, format="parquet")
    return dataset.to_batches(columns=["text"], batch_size=1024), dataset.count_rows()

def dump_one(txt: str):
    raw = txt.encode("utf-8")
    p   = OUTPUT_DIR / str(len(raw))
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{sha256(raw).hexdigest()}.txt").write_bytes(raw)
def dump_few(ls: list[str]): [dump_one(s) for s in ls]
def dump_fineweb_to_disk():
    batches, total = grab_fineweb()
    import queue
    import threading
    
    def producer(q, progress_bar):
        """Load batches one at a time, put individual texts in queue"""
        for batch in batches:
            texts = batch['text'].to_pylist()
            for text in texts:
                q.put(text)
            progress_bar.update(1)
            del texts, batch  # explicit cleanup
        # Signal end to consumers
        for _ in range(4):  # number of consumer threads
            q.put(None)
    
    def consumer(q):
        """Take texts from queue and process them"""
        while True:
            text = q.get()
            if text is None:  # end signal
                q.task_done()
                break
            dump_one(text)
            q.task_done()
    
    # Create bounded queue to limit memory usage
    q = queue.Queue(maxsize=50)  # max 50 documents buffered
    
    # Create progress bar
    progress_bar = tqdm(total=total/1024, desc="Processing batches")
    
    # Start producer and consumers
    producer_thread = threading.Thread(target=producer, args=(q, progress_bar))
    consumer_threads = [threading.Thread(target=consumer, args=(q,)) for _ in range(4)]
    
    producer_thread.start()
    for t in consumer_threads:
        t.start()
    
    # Wait for completion
    producer_thread.join()
    q.join()  # wait for all tasks to be done
    for t in consumer_threads:
        t.join()
    
    progress_bar.close()


def iter_fineweb_by_seqlen():
    for folder in sorted(OUTPUT_DIR.iterdir(), key=lambda p: int(p.name)):
        yield from sorted(folder.iterdir(), key=lambda p: p.stem)

def seqlen_sorted_fineweb(rank: int, wsize: int, msl: int=1<<15, bos: bytes=b'\xfe'):
    # Simple generator - yields lists of fineweb documents (modulo rank), where the total bytes per yield <= msl
    seqs,plen = [],0
    for i,p in enumerate(iter_fineweb_by_seqlen()):
        slen = int(p.parent.name)
        if slen > msl: return # hopeless after here
        if i%wsize != rank: continue # skip

        bstr = bos + p.read_bytes()
        blen = len(bstr)
        if plen+blen > msl:
            yield seqs
            seqs,plen = [bstr],blen
        else:
            seqs.append(bstr)
            plen += blen


# There are no checks here for partial termination. Nuke OUTPUT_DIR if you need to retry.
if __name__ == '__main__':
    if not any(OUTPUT_DIR.iterdir()): dump_fineweb_to_disk()
    # show first sample && show speed for 1000 batches
    for i,ls in enumerate(tqdm(seqlen_sorted_fineweb(1,8))):
        if not i: print(ls)
        if i > 1000: exit()
