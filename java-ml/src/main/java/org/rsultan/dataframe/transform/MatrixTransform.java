package org.rsultan.dataframe.transform;

import java.io.Serializable;
import java.util.Map;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframe.Result;
import org.rsultan.dataframe.engine.label.LabelValueIndexer;

public interface MatrixTransform extends Serializable {

  INDArray toMatrix();

  INDArray toMatrix(Map<Object, LabelValueIndexer<?>> labelValueIndexerMap);

  Result<INDArray> toMatrixResult();

  Result<INDArray> toMatrixResult(Map<Object, LabelValueIndexer<?>> objectLabelValueIndexerMap);

  Dataframe oneHotEncode(String columnToEncode);

  INDArray[] trainTest(double trainPercentage);

  INDArray[] trainTest(double trainPercentage, Map<Object, LabelValueIndexer<?>> labelValueIndexerMap);

}
