package org.regression;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface Regression {

    Regression train(INDArray X, INDArray Y);

    INDArray predict(INDArray X);
}
