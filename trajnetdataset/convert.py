"""Create Trajnet data from original datasets."""

import subprocess

import pysparkling
import trajnettools

from . import readers
from .scene import Scenes


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


def edinburgh(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .zipWithIndex()
            .flatMap(readers.edinburgh)
            .cache())


def syi(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .flatMap(readers.syi)
            .cache())


def write(input_rows, output_file, train_fraction=0.6, val_fraction=0.2):
    frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()))
    train_split_index = int(len(frames) * train_fraction)
    val_split_index = train_split_index + int(len(frames) * val_fraction)
    train_frames = set(frames[:train_split_index])
    val_frames = set(frames[train_split_index:val_split_index])
    test_frames = set(frames[val_split_index:])

    # train dataset
    train_rows = input_rows.filter(lambda r: r.frame in train_frames)
    train_output = output_file.format(split='train')
    train_scenes = Scenes().rows_to_file(train_rows, train_output)

    # validation dataset
    val_rows = input_rows.filter(lambda r: r.frame in val_frames)
    val_output = output_file.format(split='val')
    val_scenes = Scenes(start_scene_id=train_scenes.scene_id).rows_to_file(val_rows, val_output)

    # test dataset
    test_rows = input_rows.filter(lambda r: r.frame in test_frames)
    test_output = output_file.format(split='test')
    test_scenes = Scenes(start_scene_id=val_scenes.scene_id).rows_to_file(test_rows, test_output)


def main():
    sc = pysparkling.Context()

    # new datasets
    write(syi(sc, 'data/raw/syi/0?????.txt'),
          'output/{split}/syi.ndjson')
    # write(edinburgh(sc, 'data/raw/edinburgh/tracks.*.zip'),
    #       'output/{split}/edinburgh.ndjson')

    # originally train
    write(biwi(sc, 'data/raw/biwi/seq_hotel/obsmat.txt'),
          'output/{split}/biwi_hotel.ndjson')
    # write(crowds(sc, 'data/raw/crowds/arxiepiskopi1.vsp'),
    #       'output/{split}/crowds_arxiepiskopi1.ndjson')
    write(crowds(sc, 'data/raw/crowds/crowds_zara02.vsp'),
          'output/{split}/crowds_zara02.ndjson')
    write(crowds(sc, 'data/raw/crowds/crowds_zara03.vsp'),
          'output/{split}/crowds_zara03.ndjson')
    write(crowds(sc, 'data/raw/crowds/students001.vsp'),
          'output/{split}/crowds_students001.ndjson')
    write(crowds(sc, 'data/raw/crowds/students003.vsp'),
          'output/{split}/crowds_students003.ndjson')
    # write(mot(sc, 'data/raw/mot/pets2009_s2l1.txt'),
    #       'output/{split}/mot_pets2009_s2l1.ndjson')

    # originally test
    write(biwi(sc, 'data/raw/biwi/seq_eth/obsmat.txt'),
          'output/{split}/biwi_eth.ndjson')
    write(crowds(sc, 'data/raw/crowds/crowds_zara01.vsp'),
          'output/{split}/crowds_zara01.ndjson')
    write(crowds(sc, 'data/raw/crowds/uni_examples.vsp'),
          'output/{split}/crowds_uni_examples.ndjson')

    # compress the outputs
    subprocess.check_output(['tar', '-czf', 'output/test.tar.gz', 'output/test'])
    subprocess.check_output(['tar', '-czf', 'output/train.tar.gz', 'output/train'])
    subprocess.check_output(['tar', '-czf', 'output/val.tar.gz', 'output/val'])


if __name__ == '__main__':
    main()
