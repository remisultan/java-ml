package org.rsultan.core.clustering.kmedoids.strategy;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.clustering.centroid.MedoidFactory;

public enum InitialisationStrategy {
  RANDOM, PLUS_PLUS;

  public INDArray initialiseCenters(
      long k, INDArray X,
      MedoidFactory medoidFactory
  ) {
    return switch (this) {
      case RANDOM -> new RandomStrategy().initialiseCenters(k, X);
      case PLUS_PLUS -> new PlusPlusStrategy(medoidFactory).initialiseCenters(k, X);
      default -> throw new IllegalStateException("Unexpected value: " + this);
    };
  }

}
