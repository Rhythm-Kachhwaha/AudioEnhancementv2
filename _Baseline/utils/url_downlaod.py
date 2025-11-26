import requests
from tqdm import tqdm


url = "https://www.openslr.org/resources/28/rirs_noises.zip"
r = requests.get(url=url , stream=True)

size = int(r.headers.get("content-length", 0))
block_size = 128

# with tqdm(total=size , unit="B" , unit_scale=True) as progress_bar:
with open("data/rirs_noises.zip","wb") as gz , tqdm(desc="data/rirs_noises.zip" , total=size , unit='iB' , unit_scale=True , unit_divisor=128) as progress_bar:
    for chunk in r.iter_content(chunk_size=block_size):
        if chunk:
            progress_bar.update(len(chunk))
            gz.write(chunk)

if size !=0 and progress_bar.n != size:
    raise(RuntimeError("Could not donwload fiel"))


