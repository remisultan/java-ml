package org.rsultan.dataframe.transform.matrix;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.dataframe.Dataframe;

public interface MatrixTransform {

  INDArray toVector(String columnName);

  INDArray toMatrix(String... columnNames);

  Dataframe oneHotEncode(String columnToEncode);

}
