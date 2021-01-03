package org.rsultan.core.clustering.kmedoids;

import static org.rsultan.core.clustering.kmedoids.type.KMedoidType.K_MEANS;

public class KMeans extends KMedoids {

  public KMeans(int k, int numberOfIterations) {
    super(k, numberOfIterations, K_MEANS);
  }
}
