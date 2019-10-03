"""Create Trajnet data from original datasets."""

import subprocess

import pysparkling
import scipy.io
import trajnettools

from . import readers
from .scene import Scenes
from .get_type import trajectory_type


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

def write(input_rows, output_file, train_fraction=0.6, val_fraction=0.2, fps=2.5, order_frames=False):
    print(" Entering Writing ")
    ## To handle two different time stamps 7:00 and 17:00 of cff
    if order_frames:
        frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()), key=lambda frame: frame % 100000)
    else:
        frames = sorted(set(input_rows.map(lambda r: r.frame).toLocalIterator()))
    # split
    train_split_index = int(len(frames) * train_fraction)
    val_split_index = train_split_index + int(len(frames) * val_fraction)
    train_frames = set(frames[:train_split_index])
    val_frames = set(frames[train_split_index:val_split_index])
    test_frames = set(frames[val_split_index:])

    # train dataset
    train_rows = input_rows.filter(lambda r: r.frame in train_frames)
    train_output = output_file.format(split='train')
    train_scenes = Scenes(fps=fps).rows_to_file(train_rows, train_output)

    # validation dataset
    val_rows = input_rows.filter(lambda r: r.frame in val_frames)
    val_output = output_file.format(split='val')
    val_scenes = Scenes(start_scene_id=train_scenes.scene_id, fps=fps).rows_to_file(val_rows, val_output)

    # public test dataset
    test_rows = input_rows.filter(lambda r: r.frame in test_frames)
    test_output = output_file.format(split='test')
    test_scenes = Scenes(start_scene_id=val_scenes.scene_id, chunk_size=21, visible_chunk=9, fps=fps) # !!! Chunk Stride
    test_scenes.rows_to_file(test_rows, test_output)
    # private test dataset
    private_test_output = output_file.format(split='test_private')
    private_test_scenes = Scenes(start_scene_id=val_scenes.scene_id, chunk_size=21, fps=fps)
    private_test_scenes.rows_to_file(test_rows, private_test_output)

def categorize(sc, input_file, fps=2.5, train=False, test=False):
    print(" Entering Trajectory Type ")

    # Decide which folders to categorize #
    if train:
        #Train
        print("Only train")
        train_rows = get_trackrows(sc, input_file.replace('split', '').format('train'))    
        train_id  = trajectory_type(train_rows, input_file.replace('split', '').format('train'), fps=fps, track_id=0)

    elif test:
        #Test
        print("Only test")
        test_rows = get_trackrows(sc, input_file.replace('split', '').format('test_private'))    
        test_id  = trajectory_type(test_rows, input_file.replace('split', '').format('test_private'), fps=fps, track_id=0)

    else: 
        print("All Three")
        #Train
        train_rows = get_trackrows(sc, input_file.replace('split', '').format('train'))
        train_id = trajectory_type(train_rows, input_file.replace('split', '').format('train'), fps=fps, track_id=0)

        #Val
        val_rows = get_trackrows(sc, input_file.replace('split', '').format('val'))
        val_id   = trajectory_type(val_rows, input_file.replace('split', '').format('val'), fps=fps, track_id=train_id)

        #Test
        test_rows = get_trackrows(sc, input_file.replace('split', '').format('test_private'))    
        test_id  = trajectory_type(test_rows, input_file.replace('split', '').format('test_private'), fps=fps, track_id=val_id)

def main():
    sc = pysparkling.Context()

    # Example Conversions
    # # real datasets  
    write(biwi(sc, 'data/raw/biwi/seq_hotel/obsmat.txt'),
          'output_pre/{split}/biwi_hotel.ndjson')
    categorize(sc, 'output_pre/{split}/biwi_hotel.ndjson')
    write(crowds(sc, 'data/raw/crowds/crowds_zara02.vsp'),
          'output_pre/{split}/crowds_zara02.ndjson')
    categorize(sc, 'output_pre/{split}/crowds_zara02.ndjson')
    write(crowds(sc, 'data/raw/crowds/crowds_zara03.vsp'),
          'output_pre/{split}/crowds_zara03.ndjson')
    categorize(sc, 'output_pre/{split}/crowds_zara03.ndjson')
    write(crowds(sc, 'data/raw/crowds/students001.vsp'),
          'output_pre/{split}/crowds_students001.ndjson')
    categorize(sc, 'output_pre/{split}/crowds_students001.ndjson')
    write(crowds(sc, 'data/raw/crowds/students003.vsp'),
          'output_pre/{split}/crowds_students003.ndjson')
    categorize(sc, 'output_pre/{split}/crowds_students003.ndjson')

    # # new datasets
    write(wildtrack(sc, 'data/raw/wildtrack/Wildtrack_dataset/annotations_positions/*.json'),
          'output_pre/{split}/wildtrack.ndjson',fps = 2)
    categorize(sc, 'output_pre/{split}/wildtrack.ndjson',fps = 2)
    write(lcas(sc, 'data/raw/lcas/test/data.csv'),
          'output_pre/{split}/lcas.ndjson')
    categorize(sc, 'output_pre/{split}/lcas.ndjson')
    
    # # CFF: More trajectories
    # # Chunk_stride > 20 preferred. 
    # write(cff(sc, 'data/raw/cff_dataset/al_position2013-02-10.csv'),
    #       'output_pre/{split}/cff_10.ndjson', order_frames=True)  
    # categorize(sc, 'output_pre/{split}/cff_10.ndjson')
    # write(cff(sc, 'data/raw/cff_dataset/al_position2013-02-06.csv'),
    #       'output_pre/{split}/cff_06.ndjson', order_frames=True)  
    # categorize(sc, 'output_pre/{split}/cff_06.ndjson')


    # Eg. of 'Train Only' categorization
    # write(biwi(sc, 'data/raw/biwi/seq_hotel/obsmat.txt'),
    #       'output_pre/{split}/biwi_hotel.ndjson', train_fraction=1.0, val_fraction=0)
    # categorize(sc, 'output_pre/{split}/biwi_hotel.ndjson', train=True)

    # Eg. of 'Test Only' categorization
    # write(biwi(sc, 'data/raw/biwi/seq_hotel/obsmat.txt'),
    #       'output_pre/{split}/biwi_hotel.ndjson', train_fraction=0.0, val_fraction=0)
    # categorize(sc, 'output_pre/{split}/biwi_hotel.ndjson', test=True)

    # CA 
    # Generate Trajectories First. # 
    # # Train
    # write(controlled(sc, 'data/raw/controlled/orca_traj_close.txt'),
    #       'output_pre/{split}/controlled_close.ndjson', train_fraction=1.0, val_fraction=0)  
    # categorize(sc, 'output_pre/{split}/controlled_close.ndjson', train=True)
    # write(controlled(sc, 'data/raw/controlled/orca_traj_medium1.txt'),
    #       'output_pre/{split}/controlled_medium1.ndjson', train_fraction=1.0, val_fraction=0)  
    # categorize(sc, 'output_pre/{split}/controlled_medium1.ndjson', train=True)
    # write(controlled(sc, 'data/raw/controlled/orca_traj_medium2.txt'),
    #       'output_pre/{split}/controlled_medium2.ndjson', train_fraction=1.0, val_fraction=0) 
    # categorize(sc, 'output_pre/{split}/controlled_medium2.ndjson', train=True)
    # write(controlled(sc, 'data/raw/controlled/orca_traj_far.txt'),
    #       'output_pre/{split}/controlled_far.ndjson', train_fraction=1.0, val_fraction=0)            
    # categorize(sc, 'output_pre/{split}/controlled_far.ndjson', train=True)


if __name__ == '__main__':
    main()
