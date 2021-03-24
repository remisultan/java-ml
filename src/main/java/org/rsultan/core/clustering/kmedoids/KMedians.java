package org.rsultan.core.clustering.kmedoids;

import static org.rsultan.core.clustering.type.MedoidType.MEDIAN;

import org.rsultan.dataframe.Dataframe;

public class KMedians extends KMedoids {

  public KMedians(int k, int numberOfIterations) {
    super(k, numberOfIterations, MEDIAN);
  }

  @Override
  public KMedians train(Dataframe dataframe) {
    super.train(dataframe);
    return this;
  }
}
