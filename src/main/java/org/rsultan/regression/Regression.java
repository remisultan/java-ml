package org.rsultan.regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.dataframe.Dataframe;

public interface Regression {

    Regression train(Dataframe dataframe);

    Dataframe predict(Dataframe dataframe);

    INDArray computeNullHypothesis(INDArray X, INDArray W);

    double computeLoss(INDArray prediction);
}
