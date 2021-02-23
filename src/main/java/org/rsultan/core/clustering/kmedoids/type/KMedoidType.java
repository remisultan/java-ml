package org.rsultan.core.clustering.kmedoids.type;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.clustering.kmedoids.centroid.KMeansFactory;
import org.rsultan.core.clustering.kmedoids.centroid.KMediansFactory;
import org.rsultan.core.clustering.kmedoids.centroid.MedoidFactory;

public enum KMedoidType {
  K_MEANS, K_MEDIAN;

  public MedoidFactory getMedoidFactory() {
    return switch (this) {
      case K_MEANS -> new KMeansFactory();
      case K_MEDIAN -> new KMediansFactory();
    };
  }
}
