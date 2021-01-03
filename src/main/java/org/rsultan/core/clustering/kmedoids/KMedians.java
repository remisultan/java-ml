package org.rsultan.core.clustering.kmedoids;

import static org.rsultan.core.clustering.kmedoids.type.KMedoidType.K_MEDIAN;

public class KMedians extends AbstractKMedoid {

  public KMedians(int k, int numberOfIterations) {
    super(k, numberOfIterations, K_MEDIAN);
  }
}
