"""Create Trajnet data from original datasets."""

import subprocess

import pysparkling
import trajnettools

from . import readers
from .scene import to_scenes


def biwi(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.biwi)
            .cache())


def crowds(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .values()
            .flatMap(readers.crowds)
            .cache())


def mot(sc, input_file):
    """Was 7 frames per second in original recording."""
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.mot)
            .filter(lambda r: r.frame % 2 == 0)
            .cache())


def write(input_rows, output_file, train_fraction=0.6, val_fraction=0.2):
    frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()))
    train_split_index = int(len(frames) * train_fraction)
    val_split_index = train_split_index + int(len(frames) * val_fraction)
    train_frames = set(frames[:train_split_index])
    val_frames = set(frames[train_split_index:val_split_index])
    test_frames = set(frames[val_split_index:])

    # create a unique scene id across all splits
    scene_id = 0

    # train dataset
    train_rows = input_rows.filter(lambda r: r.frame in train_frames)
    for ped_id, rows in to_scenes(train_rows):
        (rows
         .map(lambda r: trajnettools.writers.trajnet(r, ped_id))
         .saveAsTextFile(output_file.format(split='train', scene_id=scene_id)))
        scene_id += 1

    # validation dataset
    val_rows = input_rows.filter(lambda r: r.frame in val_frames)
    for ped_id, rows in to_scenes(val_rows):
        (rows
         .map(lambda r: trajnettools.writers.trajnet(r, ped_id))
         .saveAsTextFile(output_file.format(split='val', scene_id=scene_id)))
        scene_id += 1

    # test dataset
    test_rows = input_rows.filter(lambda r: r.frame in test_frames)
    for ped_id, rows in to_scenes(test_rows):
        (rows
         .map(lambda r: trajnettools.writers.trajnet(r, ped_id))
         .saveAsTextFile(output_file.format(split='test', scene_id=scene_id)))
        scene_id += 1


def main():
    sc = pysparkling.Context()

    # originally train
    write(biwi(sc, 'data/raw/biwi/seq_hotel/obsmat.txt'),
          'output/{split}/biwi_hotel/{scene_id}.txt')
#     write(crowds(sc, 'data/raw/crowds/arxiepiskopi1.vsp'),
#           'output/{split}/crowds_arxiepiskopi1/{scene_id}.txt')
    write(crowds(sc, 'data/raw/crowds/crowds_zara02.vsp'),
          'output/{split}/crowds_zara02/{scene_id}.txt')
    write(crowds(sc, 'data/raw/crowds/crowds_zara03.vsp'),
          'output/{split}/crowds_zara03/{scene_id}.txt')
    write(crowds(sc, 'data/raw/crowds/students001.vsp'),
          'output/{split}/crowds_students001/{scene_id}.txt')
    write(crowds(sc, 'data/raw/crowds/students003.vsp'),
          'output/{split}/crowds_students003/{scene_id}.txt')
#     write(mot(sc, 'data/raw/mot/pets2009_s2l1.txt'),
#           'output/{split}/mot_pets2009_s2l1/{scene_id}.txt')

    # originally test
    write(biwi(sc, 'data/raw/biwi/seq_eth/obsmat.txt'),
          'output/{split}/biwi_eth/{scene_id}.txt')
    write(crowds(sc, 'data/raw/crowds/crowds_zara01.vsp'),
          'output/{split}/crowds_zara01/{scene_id}.txt')
    write(crowds(sc, 'data/raw/crowds/uni_examples.vsp'),
          'output/{split}/crowds_uni_examples/{scene_id}.txt')

    # compress the outputs
    subprocess.check_output(['tar', '-czf', 'output/test.tar.gz', 'output/test'])
    subprocess.check_output(['tar', '-czf', 'output/train.tar.gz', 'output/train'])
    subprocess.check_output(['tar', '-czf', 'output/val.tar.gz', 'output/val'])


if __name__ == '__main__':
    main()
