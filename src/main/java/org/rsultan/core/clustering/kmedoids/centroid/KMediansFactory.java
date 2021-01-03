package org.rsultan.core.clustering.kmedoids.centroid;

import static org.nd4j.linalg.ops.transforms.Transforms.allManhattanDistances;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public class KMediansFactory implements MedoidFactory {

  @Override
  public INDArray computeDistance(INDArray centroids, INDArray vector) {
    return allManhattanDistances(centroids, vector, 1);
  }

  @Override
  public INDArray computeMedoids(INDArray assignedCentroids) {
    return assignedCentroids.median(0);
  }
}
