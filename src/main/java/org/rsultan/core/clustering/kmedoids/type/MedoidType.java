package org.rsultan.core.clustering.kmedoids.type;

import org.rsultan.core.clustering.kmedoids.centroid.MeanFactory;
import org.rsultan.core.clustering.kmedoids.centroid.MedianFactory;
import org.rsultan.core.clustering.kmedoids.centroid.MedoidFactory;

public enum MedoidType {
  MEAN, MEDIAN;

  public MedoidFactory getMedoidFactory() {
    return switch (this) {
      case MEAN -> new MeanFactory();
      case MEDIAN -> new MedianFactory();
    };
  }
}
