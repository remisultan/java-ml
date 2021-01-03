package org.rsultan.core.clustering.kmedoids.centroid;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.INDArrayIndex;

public interface MedoidFactory {

  INDArray computeDistance(INDArray centroids, INDArray vector);

  INDArray computeMedoids(INDArray assignedCentroids);

}
