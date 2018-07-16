"""Trying to reproduce original Trajnet dataset."""

import pysparkling
import trajnetdataset
import trajnettools


def main():
    sc = pysparkling.Context()

    biwi_train = (sc
                  .textFile('data/raw/biwi/seq_hotel/obsmat.txt')
                  .map(trajnetdataset.readers.biwi)
                  .cache())

    good_start_frames = set(biwi_train
                            .groupBy(lambda r: r.pedestrian)
                            .filter(lambda kv: len(kv[1]) >= 20)
                            .values()
                            .map(lambda rs: rs[0].frame)
                            .collect())

    # good_start_frames_filtered = []
    # for f in sorted(good_start_frames):
    #     if good_start_frames_filtered and \
    #        f <= good_start_frames_filtered[-1] + 20:
    #         continue
    #     good_start_frames_filtered.append(f)
    # print(len(good_start_frames), len(good_start_frames_filtered))
    # print(good_start_frames_filtered)
    good_start_frames_filtered = good_start_frames

    good_frames = {f
                   for s in good_start_frames_filtered
                   for f in range(s, s + 200, 10)}
    print(sorted(good_frames))

    (biwi_train
     .filter(lambda r: r.frame in good_frames)

     # filter out short pedestrian paths
     .groupBy(lambda r: r.pedestrian)
     .filter(lambda kv: len(kv[1]) >= 20)
     .mapValues(lambda rs: rs[:20])
     .values()
     .flatMap(lambda v: v)

     # write output
     .sortBy(lambda r: (r.pedestrian, r.frame))
     .map(trajnettools.writers.trajnet_tracks)
     .saveAsTextFile('data/train/biwi/biwi_hotel.ndjson'))


if __name__ == '__main__':
    main()
