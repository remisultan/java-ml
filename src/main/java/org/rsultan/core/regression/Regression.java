package org.rsultan.core.regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.Trainable;
import org.rsultan.dataframe.Dataframe;

public interface Regression extends Trainable<Regression> {

  Dataframe predict(Dataframe dataframe);

  INDArray computeNullHypothesis(INDArray X, INDArray W);

  double computeLoss(INDArray prediction);

}
