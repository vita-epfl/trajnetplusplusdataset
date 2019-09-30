Install
-------

.. code-block:: sh

    pip install -e '.[test,plot]'
    pylint trajnetdataset
    pytest


Prepare Data
------------

Existing data:

.. code-block::

    data/
        data_arxiepiskopi.rar
        data_university_students.rar
        data_zara.rar
        ewap_dataset_light.tgz
        # 3DMOT2015Labels  # from: https://motchallenge.net/data/3DMOT2015Labels.zip (video file at http://cs.binghamton.edu/~mrldata/public/PETS2009/S2_L1.tar.bz2)
        Train.zip  # from trajnet.epfl.ch
        cvpr2015_pedestrianWalkingPathDataset.rar  # from http://www.ee.cuhk.edu.hk/~syi/ (website not accessible but data are also here: https://www.dropbox.com/s/7y90xsxq0l0yv8d/cvpr2015_pedestrianWalkingPathDataset.rar?dl=0.+63)
        cff_dataset.zip # from https://www.dropbox.com/s/cnnk2ofreeoshuz/cff_dataset.zip?dl=0

Extract:

.. code-block:: sh

    # biwi
    mkdir -p data/raw/biwi
    tar -xzf data/ewap_dataset_light.tgz --strip-components=1 -C data/raw/biwi

    # crowds
    mkdir -p data/raw/crowds
    unrar e data/data_arxiepiskopi.rar data/raw/crowds
    unrar e data/data_university_students.rar data/raw/crowds
    unrar e data/data_zara.rar data/raw/crowds

    # cff
    mkdir -p data/raw/cff_dataset
    unzip data/cff_dataset.zip -d data/raw/
    rm -r data/raw/__MACOSX

    # Wildtrack: https://www.epfl.ch/labs/cvlab/data/data-wildtrack/
    mkdir -p data/raw/wildtrack
    unzip data/Wildtrack_dataset_full.zip -d data/raw/wildtrack

    # L-CAS: https://drive.google.com/drive/folders/1CPV9XeJsZzvtTxPQ9u1ppLGs_29e-XdQ
    mkdir -p data/raw/lcas
    cp data/lcas_pedestrian_dataset/minerva/train/data.csv data/raw/lcas

    # pedestrian walking dataset
    mkdir -p data/raw/syi
    unrar e data/cvpr2015_pedestrianWalkingPathDataset.rar data/raw/syi

    PETS09 S2L1 ground truth -- not used because people behavior is not normal
    mkdir -p data/raw/mot
    unzip data/3DMOT2015Labels.zip -d data/
    cp data/3DMOT2015Labels/train/PETS09-S2L1/gt/gt.txt data/raw/mot/pets2009_s2l1.txt

    # Edinburgh Informatics Forum tracker -- not used because tracks are not good enough
    mkdir -p data/raw/edinburgh
    wget -i edinburgh_informatics_forum_urls.txt -P data/raw/edinburgh/

Run
---

.. code-block:: sh

    python -m trajnetdataset.convert

    # create plots to check new dataset
    python -m trajnettools.summarize output/train/*.ndjson


Difference in generated data
----------------------------

* partial tracks are now included (for correct occupancy maps)
* pedestrians that appear in multiple chunks had the same id before (might be a problem for some input readers)
* explicit index of scenes with annotation of the primary pedestrian

# * the primary pedestrian has to move by more than 1 meter
* at one point, the primary pedestrian has to be <3m away from another pedestrian

Citations
---------

* ``syi``: Shuai Yi, Hongsheng Li, and Xiaogang Wang. Understanding Pedestrian Behaviors from Stationary Crowd Groups. In Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2015).
* ``edinburgh``: B. Majecka, "Statistical models of pedestrian behaviour in the Forum", MSc Dissertation, School of Informatics, University of Edinburgh, 2009.
* ``wildtrack``:

.. code-block::

    @inproceedings{chavdarova-et-al-2018,
        author = "Chavdarova, T. and Baqué, P. and Bouquet, S. and Maksai, A. and Jose, C. and Bagautdinov, T. and Lettry, L. and Fua, P. and Van Gool, L. and Fleuret, F.",
        title = {{WILDTRACK}: A Multi-camera {HD} Dataset for Dense Unscripted Pedestrian Detection},
        journal = "Proceedings of the IEEE international conference on Computer Vision and Pattern Recognition (CVPR)",
        year = 2018,
    }
