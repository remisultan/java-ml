package org.rsultan.core;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.dataframe.Dataframe;

public interface Trainable<T> {

    T train(Dataframe dataframe);

}
