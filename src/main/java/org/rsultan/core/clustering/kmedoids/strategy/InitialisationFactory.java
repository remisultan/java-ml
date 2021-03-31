package org.rsultan.core.clustering.kmedoids.strategy;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.clustering.centroid.MedoidFactory;

public interface InitialisationFactory {

  INDArray initialiseCenters(long K, INDArray X);

}
