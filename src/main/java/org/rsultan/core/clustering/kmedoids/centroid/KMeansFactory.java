package org.rsultan.core.clustering.kmedoids.centroid;

import static org.nd4j.linalg.ops.transforms.Transforms.allEuclideanDistances;

import org.nd4j.linalg.api.ndarray.INDArray;

public class KMeansFactory implements MedoidFactory {

  @Override
  public INDArray computeDistance(INDArray centroids, INDArray vector) {
    return allEuclideanDistances(centroids, vector, 1);
  }

  @Override
  public INDArray computeMedoids(INDArray assignedCentroids) {
    return assignedCentroids.mean(true, 0);
  }
}
