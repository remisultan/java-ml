package org.rsultan.core.clustering.kmedoids;

import static org.rsultan.core.clustering.kmedoids.type.MedoidType.MEDIAN;

public class KMedians extends KMedoids {

  public KMedians(int k, int numberOfIterations) {
    super(k, numberOfIterations, MEDIAN);
  }
}
