package org.rsultan.core.clustering.kmedoids;

import static org.rsultan.core.clustering.kmedoids.strategy.InitialisationStrategy.PLUS_PLUS;
import static org.rsultan.core.clustering.type.MedoidType.MEAN;

import org.rsultan.core.clustering.kmedoids.strategy.InitialisationStrategy;
import org.rsultan.dataframe.Dataframe;

public class KMeans extends KMedoids {

  public KMeans(int k, int numberOfIterations) {
    this(k, numberOfIterations, PLUS_PLUS);
  }
  public KMeans(int k, int numberOfIterations, InitialisationStrategy strategy) {
    super(k, numberOfIterations, MEAN, strategy);
  }

  @Override
  public KMeans train(Dataframe dataframe) {
    super.train(dataframe);
    return this;
  }

  @Override
  public String toString() {
    return "KMeans{" +
        "K=" + K +
        ", medoidType=" + medoidType +
        ", numberOfIterations=" + numberOfIterations +
        ", initialisationStrategy=" + initialisationStrategy +
        '}';
  }
}
