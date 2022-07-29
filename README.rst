Generating Toy Synthetic Datasets
---------------------------------

Install
-------

.. code-block:: sh

    pip install -e '.[test,plot]'
    sh setup_orca.sh
    sh setup_social_force.sh


Prepare Two-Ped Dataset: Approaching Head-On
--------------------------------------------

.. code-block:: sh

    # Generate ORCA
   	python -m trajnetdataset.controlled_data --mode trajnet --num_scenes 1000 --simulation_scene 'two_ped' --num_ped 2
	# Convert
    python -m trajnetdataset.convert --direct --synthetic --mode trajnet --linear_threshold 0.3 --acceptance 0.0 0.0 1.0 0.0 --orca_file data/raw/controlled/orca_two_ped_2ped_1000scenes_.txt --goal_file goal_files/train/orca_two_ped_2ped_1000scenes_.pkl --output_filename orca_synthetic_two_ped_1000


Prepare Two-Ped Dataset: Approach At Different Angles
-----------------------------------------------------

.. code-block:: sh

    # Generate ORCA
   	python -m trajnetdataset.controlled_data --mode trajnet --num_scenes 1000 --simulation_scene 'two_ped_angle' --num_ped 2
	# Convert
    python -m trajnetdataset.convert --direct --synthetic --mode trajnet --linear_threshold 0.3 --acceptance 0.0 0.0 1.0 0.0 --orca_file data/raw/controlled/orca_two_ped_angle_2ped_1000scenes_.txt --goal_file goal_files/train/orca_two_ped_angle_2ped_1000scenes_.pkl --output_filename orca_synthetic_two_ped_angle_1000



Preparing Three-Ped Dataset
---------------------------

.. code-block:: sh

    # Generate ORCA
   	python -m trajnetdataset.controlled_data --mode trajnet --num_scenes 1000 --num_ped 3
	# Convert
    python -m trajnetdataset.convert --direct --synthetic --mode trajnet --linear_threshold 0.3 --acceptance 0.0 0.0 1.0 0.0 --orca_file data/raw/controlled/orca_circle_crossing_3ped_1000scenes_.txt --goal_file goal_files/train/orca_circle_crossing_3ped_1000scenes_.pkl --output_filename orca_synthetic_three_ped_1000


Difference in generated data in TrajNet++
-----------------------------------------

* partial tracks are now included (for correct occupancy maps)
* pedestrians that appear in multiple chunks had the same id before (might be a problem for some input readers)
* explicit index of scenes with annotation of the primary pedestrian

# * the primary pedestrian has to move by more than 1 meter
* at one point, the primary pedestrian has to be <3m away from another pedestrian

Citation
========

If you find this code useful in your research then please cite

.. code-block::
    
    @article{Kothari2020HumanTF,
      author={Kothari, Parth and Kreiss, Sven and Alahi, Alexandre},
      journal={IEEE Transactions on Intelligent Transportation Systems}, 
      title={Human Trajectory Forecasting in Crowds: A Deep Learning Perspective}, 
      year={2021},
      volume={},
      number={},
      pages={1-15},
      doi={10.1109/TITS.2021.3069362}
     }
