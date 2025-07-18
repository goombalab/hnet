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
    from multiprocessing import Pool
    g = (batch['text'].to_pylist() for batch in tqdm(batches, total=total/1024))
    with Pool(32) as p: p.map(dump_few, g)


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
