def euclidean_distance_2(row1, row2):
    return (row1.x - row2.x)**2 + (row1.y - row2.y)**2


def to_scenes(rows, chunk_size=20, chunk_stride=5):
    by_pedestrian = rows.groupBy(lambda r: r.pedestrian).cache()

    # scenes: pedestrian of interest, [(start frame, end frame)]
    scenes = (
        by_pedestrian
        .filter(lambda p_path: len(p_path[1]) >= chunk_size)
        .flatMapValues(lambda path: [
            (path[i].frame, path[i+chunk_size].frame)
            for i in range(0, len(path) - chunk_size, chunk_stride)
            # filter for pedestrians moving by more than 1 meter
            if euclidean_distance_2(path[i], path[i+chunk_size]) > 1.0
        ])
    )

    # output
    for ped_id, (start, end) in scenes.collect():
        scene_rows = (rows
                      .filter(lambda r: start <= r.frame <= end)
                      .sortBy(lambda r: (r.pedestrian, r.frame)))
        yield ped_id, scene_rows
