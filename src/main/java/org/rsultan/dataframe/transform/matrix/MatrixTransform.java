package org.rsultan.dataframe.transform.matrix;

import java.io.Serializable;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.dataframe.Dataframe;

public interface MatrixTransform extends Serializable {

  INDArray toVector(String columnName);

  INDArray toMatrix(String... columnNames);

  Dataframe oneHotEncode(String columnToEncode);

}
