"""Create Trajnet data from original datasets."""
import argparse
import shutil

import pysparkling
import scipy.io

from . import readers
from .scene import Scenes
from .get_type import trajectory_type

import warnings
warnings.filterwarnings("ignore")

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


def dukemtmc(sc, input_file):
    print('processing ' + input_file)
    contents = scipy.io.loadmat(input_file)['trainData']
    return (sc
            .parallelize(readers.dukemtmc(contents))
            .cache())


def wildtrack(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .wholeTextFiles(input_file)
            .flatMap(readers.wildtrack)
            .cache())

def cff(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.cff)
            .filter(lambda r: r is not None)
            .cache())

def lcas(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.lcas)
            .cache())

def controlled(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.controlled)
            .cache())

def get_trackrows(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.get_trackrows)
            .filter(lambda r: r is not None)
            .cache())

def standard(sc, input_file):
    print('processing ' + input_file)
    return (sc
            .textFile(input_file)
            .map(readers.standard)
            .cache())

def write(input_rows, output_file, args):
    """ Write Valid Scenes without categorization """

    print(" Entering Writing ")
    ## To handle two different time stamps 7:00 and 17:00 of cff
    if args.order_frames:
        frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()),
                        key=lambda frame: frame % 100000)
    else:
        frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()))

    # split
    train_split_index = int(len(frames) * args.train_fraction)
    val_split_index = train_split_index + int(len(frames) * args.val_fraction)
    train_frames = set(frames[:train_split_index])
    val_frames = set(frames[train_split_index:val_split_index])
    test_frames = set(frames[val_split_index:])

    # train dataset
    train_rows = input_rows.filter(lambda r: r.frame in train_frames)
    train_output = output_file.format(split='train')
    train_scenes = Scenes(fps=args.fps, start_scene_id=0, args=args).rows_to_file(train_rows, train_output)

    # validation dataset
    val_rows = input_rows.filter(lambda r: r.frame in val_frames)
    val_output = output_file.format(split='val')
    val_scenes = Scenes(fps=args.fps, start_scene_id=train_scenes.scene_id, args=args).rows_to_file(val_rows, val_output)

    # public test dataset
    test_rows = input_rows.filter(lambda r: r.frame in test_frames)
    test_output = output_file.format(split='test')
    test_scenes = Scenes(fps=args.fps, start_scene_id=val_scenes.scene_id, args=args) # !!! Chunk Stride
    test_scenes.rows_to_file(test_rows, test_output)

    # private test dataset
    private_test_output = output_file.format(split='test_private')
    private_test_scenes = Scenes(fps=args.fps, start_scene_id=val_scenes.scene_id, args=args)
    private_test_scenes.rows_to_file(test_rows, private_test_output)

def categorize(sc, input_file, args):
    """ Categorize the Scenes """

    print(" Entering Categorizing ")
    test_fraction = 1 - args.train_fraction - args.val_fraction

    train_id = 0
    if args.train_fraction:
        print("Categorizing Training Set")
        train_rows = get_trackrows(sc, input_file.replace('split', '').format('train'))
        train_id = trajectory_type(train_rows, input_file.replace('split', '').format('train'),
                                   fps=args.fps, track_id=0, args=args)

    val_id = train_id
    if args.val_fraction:
        print("Categorizing Validation Set")
        val_rows = get_trackrows(sc, input_file.replace('split', '').format('val'))
        val_id = trajectory_type(val_rows, input_file.replace('split', '').format('val'),
                                 fps=args.fps, track_id=train_id, args=args)


    if test_fraction:
        print("Categorizing Test Set")
        test_rows = get_trackrows(sc, input_file.replace('split', '').format('test_private'))
        _ = trajectory_type(test_rows, input_file.replace('split', '').format('test_private'),
                            fps=args.fps, track_id=val_id, args=args)

def edit_goal_file(old_filename, new_filename):
    """ Rename goal files. 
    The name of goal files should be identical to the data files
    """

    shutil.copy("goal_files/train/" + old_filename, "goal_files/train/" + new_filename)
    shutil.copy("goal_files/val/" + old_filename, "goal_files/val/" + new_filename)
    shutil.copy("goal_files/test_private/" + old_filename, "goal_files/test_private/" + new_filename)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', type=int, default=9,
                        help='Length of observation')
    parser.add_argument('--pred_len', type=int, default=12,
                        help='Length of prediction')
    parser.add_argument('--train_fraction', default=0.6, type=float,
                        help='Training set fraction')
    parser.add_argument('--val_fraction', default=0.2, type=float,
                        help='Validation set fraction')
    parser.add_argument('--fps', default=2.5, type=float,
                        help='fps')
    parser.add_argument('--order_frames', action='store_true',
                        help='For CFF')
    parser.add_argument('--chunk_stride', type=int, default=2,
                        help='Sampling Stride')
    parser.add_argument('--min_length', default=0.0, type=float,
                        help='Min Length of Primary Trajectory')
    parser.add_argument('--synthetic', action='store_true',
                        help='convert synthetic datasets (if false, convert real)')
    parser.add_argument('--all_present', action='store_true',
                        help='filter scenes where all pedestrians present at all times')
    parser.add_argument('--goal_file', default=None,
                        help='Pkl file for goals (required for ORCA sensitive scene filtering)')
    parser.add_argument('--mode', default='default', choices=('default', 'trajnet'),
                        help='mode of ORCA scene generation (required for ORCA sensitive scene filtering)')

    ## For Trajectory categorizing and filtering
    categorizers = parser.add_argument_group('categorizers')
    categorizers.add_argument('--static_threshold', type=float, default=1.0,
                              help='Type I static threshold')
    categorizers.add_argument('--linear_threshold', type=float, default=0.5,
                              help='Type II linear threshold (0.3 for Synthetic)')
    categorizers.add_argument('--inter_dist_thresh', type=float, default=5,
                              help='Type IIId distance threshold for cone')
    categorizers.add_argument('--inter_pos_range', type=float, default=15,
                              help='Type IIId angle threshold for cone (degrees)')
    categorizers.add_argument('--grp_dist_thresh', type=float, default=0.8,
                              help='Type IIIc distance threshold for group')
    categorizers.add_argument('--grp_std_thresh', type=float, default=0.2,
                              help='Type IIIc std deviation for group')
    categorizers.add_argument('--acceptance', nargs='+', type=float, default=[0.1, 1, 1, 1],
                              help='acceptance ratio of different trajectory (I, II, III, IV) types')

    args = parser.parse_args()
    sc = pysparkling.Context()

    # Real datasets conversion (eg. ETH)
    #########################
    ## Training Set
    #########################
    args.train_fraction = 1.0
    args.val_fraction = 0.0

    ## Biwi
    write(standard(sc, 'data/eth/train/biwi_hotel_train.txt'),
          'output_pre/{split}/biwi_hotel_train.ndjson', args)
    categorize(sc, 'output_pre/{split}/biwi_hotel_train.ndjson', args)

    ## Crowds Zara
    write(standard(sc, 'data/eth/train/crowds_zara02_train.txt '),
          'output_pre/{split}/crowds_zara02_train.ndjson', args)
    categorize(sc, 'output_pre/{split}/crowds_zara02_train.ndjson', args)

    write(standard(sc, 'data/eth/train/crowds_zara01_train.txt '),
          'output_pre/{split}/crowds_zara01_train.ndjson', args)
    categorize(sc, 'output_pre/{split}/crowds_zara01_train.ndjson', args)

    write(standard(sc, 'data/eth/train/crowds_zara03_train.txt '),
          'output_pre/{split}/crowds_zara03_train.ndjson', args)
    categorize(sc, 'output_pre/{split}/crowds_zara03_train.ndjson', args)

    ## Crowds Uni
    write(standard(sc, 'data/eth/train/students001_train.txt '),
          'output_pre/{split}/students001_train.ndjson', args)
    categorize(sc, 'output_pre/{split}/students001_train.ndjson', args)

    write(standard(sc, 'data/eth/train/uni_examples_train.txt '),
          'output_pre/{split}/uni_examples_train.ndjson', args)
    categorize(sc, 'output_pre/{split}/uni_examples_train.ndjson', args)

    write(standard(sc, 'data/eth/train/students003_train.txt '),
          'output_pre/{split}/students003_train.ndjson', args)
    categorize(sc, 'output_pre/{split}/students003_train.ndjson', args)

    #########################
    ## Validation Set
    #########################
    args.train_fraction = 0.0
    args.val_fraction = 1.0

    ## Biwi
    write(standard(sc, 'data/eth/val/biwi_hotel_val.txt'),
          'output_pre/{split}/biwi_hotel_val.ndjson', args)
    categorize(sc, 'output_pre/{split}/biwi_hotel_val.ndjson', args)

    ## Crowds Zara
    write(standard(sc, 'data/eth/val/crowds_zara02_val.txt '),
          'output_pre/{split}/crowds_zara02_val.ndjson', args)
    categorize(sc, 'output_pre/{split}/crowds_zara02_val.ndjson', args)

    write(standard(sc, 'data/eth/val/crowds_zara01_val.txt '),
          'output_pre/{split}/crowds_zara01_val.ndjson', args)
    categorize(sc, 'output_pre/{split}/crowds_zara01_val.ndjson', args)

    write(standard(sc, 'data/eth/val/crowds_zara03_val.txt '),
          'output_pre/{split}/crowds_zara03_val.ndjson', args)
    categorize(sc, 'output_pre/{split}/crowds_zara03_val.ndjson', args)

    ## Crowds Uni
    write(standard(sc, 'data/eth/val/students001_val.txt '),
          'output_pre/{split}/students001_val.ndjson', args)
    categorize(sc, 'output_pre/{split}/students001_val.ndjson', args)

    write(standard(sc, 'data/eth/val/uni_examples_val.txt '),
          'output_pre/{split}/uni_examples_val.ndjson', args)
    categorize(sc, 'output_pre/{split}/uni_examples_val.ndjson', args)

    write(standard(sc, 'data/eth/val/students003_val.txt '),
          'output_pre/{split}/students003_val.ndjson', args)
    categorize(sc, 'output_pre/{split}/students003_val.ndjson', args)

    #########################
    ## Testing Set
    #########################
    args.train_fraction = 0.0
    args.val_fraction = 0.0

    ## ETH
    write(standard(sc, 'data/eth/test/biwi_eth.txt '),
          'output_pre/{split}/biwi_eth.ndjson', args)
    categorize(sc, 'output_pre/{split}/biwi_eth.ndjson', args)


if __name__ == '__main__':
    main()
