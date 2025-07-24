from os import sched_getaffinity
from tqdm import tqdm
from pathlib import Path
from hashlib import sha256
from pyarrow import dataset as ds
from huggingface_hub import snapshot_download
from queue import Queue
from threading import Thread

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
def dump_fineweb_to_disk(consumers: int=len(sched_getaffinity(0))):
    batches, total = grab_fineweb()
    qsize = consumers * 4
    bsz = qsize * 16
    
    def producer(q, pbar):
        for batch in batches:
            for s in batch['text'].to_pylist(): q.put(s)
            pbar.update(1)
        for _ in range(consumers): q.put(None)
    def consumer(q):
        while text := q.get():
            dump_one(text)
            q.task_done()
        q.task_done()
    
    q = Queue(maxsize=qsize)
    progress_bar = tqdm(total=total/bsz, desc="Processing batches")

    threads = [
        Thread(target=producer, args=(q, progress_bar)), # producer
        *[Thread(target=consumer, args=(q,)) for _ in range(consumers)]
    ]
    for t in threads: t.start()
    q.join()
    for t in threads: t.join()
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
