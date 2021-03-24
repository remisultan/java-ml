package org.rsultan.core.clustering.centroid;

import static org.nd4j.linalg.ops.transforms.Transforms.allEuclideanDistances;

import org.nd4j.linalg.api.ndarray.INDArray;

public class MeanFactory implements MedoidFactory {

  @Override
  public INDArray computeDistance(INDArray centroids, INDArray vector) {
    return allEuclideanDistances(centroids, vector, 1);
  }

  @Override
  public INDArray computeMedoids(INDArray assignedCentroids) {
    return assignedCentroids.mean(true, 0);
  }

  @Override
  public double computeNorm(INDArray diff) {
    return diff.norm2Number().doubleValue();
  }
}
