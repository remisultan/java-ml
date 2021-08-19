package org.rsultan.core.clustering.type;

import org.rsultan.core.clustering.centroid.MeanFactory;
import org.rsultan.core.clustering.centroid.MedianFactory;
import org.rsultan.core.clustering.centroid.MedoidFactory;

public enum MedoidType {
  MEAN, MEDIAN;

  public MedoidFactory getMedoidFactory() {
    return switch (this) {
      case MEAN -> new MeanFactory();
      case MEDIAN -> new MedianFactory();
    };
  }
}
