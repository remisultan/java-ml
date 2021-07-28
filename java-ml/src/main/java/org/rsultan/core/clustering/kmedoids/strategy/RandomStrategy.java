package org.rsultan.core.clustering.kmedoids.strategy;

import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;
import org.nd4j.linalg.api.ndarray.INDArray;

public class RandomStrategy implements InitialisationFactory {

  @Override
  public INDArray initialiseCenters(long K, INDArray X) {
    var centers = IntStream
        .generate(() -> ThreadLocalRandom.current().nextInt(X.columns()))
        .limit(K)
        .toArray();
    return X.getColumns(centers).transpose();
  }
}
