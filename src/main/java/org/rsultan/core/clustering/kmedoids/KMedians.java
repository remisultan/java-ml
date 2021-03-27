package org.rsultan.core.clustering.kmedoids;

import static org.rsultan.core.clustering.kmedoids.strategy.InitialisationStrategy.PLUS_PLUS;
import static org.rsultan.core.clustering.type.MedoidType.MEDIAN;

import org.rsultan.core.clustering.kmedoids.strategy.InitialisationStrategy;
import org.rsultan.dataframe.Dataframe;

public class KMedians extends KMedoids {

  public KMedians(int k, int numberOfIterations) {
    this(k, numberOfIterations, PLUS_PLUS);
  }

  public KMedians(int k, int numberOfIterations, InitialisationStrategy strategy) {
    super(k, numberOfIterations, MEDIAN, strategy);
  }

  @Override
  public KMedians train(Dataframe dataframe) {
    super.train(dataframe);
    return this;
  }

  @Override
  public String toString() {
    return "KMedians{" +
        "K=" + K +
        ", medoidType=" + medoidType +
        ", numberOfIterations=" + numberOfIterations +
        ", initialisationStrategy=" + initialisationStrategy +
        '}';
  }
}
