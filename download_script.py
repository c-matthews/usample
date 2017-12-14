# script to download data files for supernova type Ia likelihood calculations
# based on the script https://stackoverflow.com/a/43958201 

import sys
try:
    import urllib.request
    python3 = True
except ImportError:
    import urllib2
    python3 = False


def progress_callback_simple(downloaded,total):
    sys.stdout.write(
        "\r" +
        (len(str(total))-len(str(downloaded)))*" " + str(downloaded) + "/%d"%total +
        " [%3.2f%%]"%(100.0*float(downloaded)/float(total))
    )
    sys.stdout.flush()

def download(srcurl, dstfilepath, progress_callback=None, block_size=8192):
    def _download_helper(response, out_file, file_size):
        if progress_callback!=None: progress_callback(0,file_size)
        if block_size == None:
            buffer = response.read()
            out_file.write(buffer)

            if progress_callback!=None: progress_callback(file_size,file_size)
        else:
            file_size_dl = 0
            while True:
                buffer = response.read(block_size)
                if not buffer: break

                file_size_dl += len(buffer)
                out_file.write(buffer)

                if progress_callback!=None: progress_callback(file_size_dl,file_size)
    with open(dstfilepath,"wb") as out_file:
        if python3:
            with urllib.request.urlopen(srcurl) as response:
                file_size = int(response.getheader("Content-Length"))
                _download_helper(response,out_file,file_size)
        else:
            response = urllib2.urlopen(srcurl)
            meta = response.info()
            file_size = int(meta.getheaders("Content-Length")[0])
            _download_helper(response,out_file,file_size)

import os
import tarfile
import zipfile

def extract_file(path, to_directory='.'):
    if path.endswith('.zip'):
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        opener, mode = tarfile.open, 'r:bz2'
    else: 
        raise ValueError, "Could not extract `%s` as no appropriate extractor is found" % path
    
    cwd = os.getcwd()
    os.chdir(to_directory)
    
    try:
        file = opener(path, mode)
        try: file.extractall()
        finally: file.close()
    finally:
        os.chdir(cwd)
   
if __name__ == '__main__':
    if len(sys.argv) > 1:
        print("will download data to directory %s"%sys.argv[1])
        data_dir = sys.argv[1]
    else:
        print("path to data directory is not supplied")
        print("usage: python download_data.py dir_for_data_download")
        exit(1)
        
    
    import traceback
    try:
        
        download(     
            "http://astro.uchicago.edu/~andrey/usample/sn_data.zip",
            data_dir+"sn_data.zip",
            progress_callback_simple
        )
        print("\n extracting...")
        extract_file(data_dir+"sn_data.zip", to_directory = data_dir)
        
        os.remove(data_dir+"sn_data.zip")
        
    except:
        traceback.print_exc()
        input()
    