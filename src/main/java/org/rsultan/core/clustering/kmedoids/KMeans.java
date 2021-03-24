package org.rsultan.core.clustering.kmedoids;

import static org.rsultan.core.clustering.type.MedoidType.MEAN;

import org.rsultan.dataframe.Dataframe;

public class KMeans extends KMedoids {

  public KMeans(int k, int numberOfIterations) {
    super(k, numberOfIterations, MEAN);
  }

  @Override
  public KMeans train(Dataframe dataframe) {
    super.train(dataframe);
    return this;
  }
}
