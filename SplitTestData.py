import os
import argparse
from os import listdir
from os.path import isfile, join
from collections import OrderedDict


def split(filepath):
    users = [id for id in listdir(filepath) if not isfile(join(filepath, id))]
    ret={}
    for id in users:
        ret[id] = ([],[])
        left=0
        right=0
        localPath = join(filepath, id)
        clips = [id for id in listdir(localPath) if not isfile(join(localPath, id))]
        d={}
        for clip in clips:
            clipsInSegment=0
            clipPath = join(localPath,clip)
            segments = [id for id in listdir(clipPath) if isfile(join(clipPath, id))]
            for segment in segments:
                clipsInSegment+=1
            if clipsInSegment in d.keys():
                d[clipsInSegment].append(clip)
            else: d[clipsInSegment]=[clip]
        od = OrderedDict(sorted(d.items(),reverse=True))
        for k,v in od.items():
            for clip in v:
                if left<=right:
                    left+=k
                    ret[id][0].append(clip)
                else:
                    right+=k
                    ret[id][1].append(clip)
    return ret


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath")
    args = parser.parse_args()
    filepath = args.filepath
    print(split(filepath))