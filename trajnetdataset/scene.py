from collections import defaultdict

import trajnettools
from trajnettools import SceneRow


class Scenes(object):
    def __init__(self, start_scene_id=0, chunk_size=20, chunk_stride=5):
        self.scene_id = start_scene_id
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.frames = set()

    @staticmethod
    def euclidean_distance_2(row1, row2):
        """Euclidean distance squared between two rows."""
        return (row1.x - row2.x)**2 + (row1.y - row2.y)**2

    @staticmethod
    def close_pedestrians(rows, cell_size=3):
        """Fast computation of spatially close pedestrians.

        By frame, get the list of pedestrian ids that or close to other
        pedestrians. Approximate with multi-occupancy of discrete grid cells.
        """
        sparse_occupancy = defaultdict(list)
        for row in rows:
            x = int(row.x // cell_size * cell_size)
            y = int(row.y // cell_size * cell_size)
            sparse_occupancy[(x, y)].append(row.pedestrian)
        return {ped_id
                for cell in sparse_occupancy.values() if len(cell) > 1
                for ped_id in cell}

    def from_rows(self, rows):
        count_by_frame = rows.groupBy(lambda r: r.frame).mapValues(len).collectAsMap()
        occupancy_by_frame = (rows
                              .groupBy(lambda r: r.frame)
                              .mapValues(self.close_pedestrians)
                              .collectAsMap())

        # scenes: pedestrian of interest, [frames]
        scenes = (
            rows
            .groupBy(lambda r: r.pedestrian)
            .filter(lambda p_path: len(p_path[1]) >= self.chunk_size)
            .flatMapValues(lambda path: [
                [path[ii].frame for ii in range(i, i + self.chunk_size + 1)]
                for i in range(0, len(path) - self.chunk_size, self.chunk_stride)
                # filter for pedestrians moving by more than 1 meter
                if self.euclidean_distance_2(path[i], path[i+self.chunk_size]) > 1.0
            ])
            .collect()
        )

        # filtered output
        filtered_scenes = []
        for ped_id, scene_frames in scenes:
            n_rows = sum(count_by_frame[f] for f in scene_frames)

            # filter for scenes that have some activity
            if n_rows < self.chunk_size * 2.0:
                continue

            # detect proximity
            if ped_id not in {p
                              for frame in scene_frames
                              for p in occupancy_by_frame[frame]}:
                continue

            # add frames
            self.frames |= set(scene_frames)

            filtered_scenes.append(
                SceneRow(self.scene_id, ped_id, scene_frames[0], scene_frames[-1]))

            self.scene_id += 1

        return rows.context.parallelize(filtered_scenes)

    def rows_to_file(self, rows, output_file):
        scenes = self.from_rows(rows)
        tracks = rows.filter(lambda r: r.frame in self.frames)
        all_data = rows.context.union((scenes, tracks))
        all_data.map(trajnettools.writers.trajnet).saveAsTextFile(output_file)

        return self
