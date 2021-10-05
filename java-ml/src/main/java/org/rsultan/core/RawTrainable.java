package org.rsultan.core;

import java.io.Serializable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.dataframe.Dataframe;

public interface RawTrainable<T> extends Serializable {

  T train(INDArray matrix);

  INDArray predict(INDArray matrix);

}
