package org.rsultan.core.tree;

import static org.rsultan.core.tree.impurity.ImpurityStrategy.RMSE;

import java.util.List;
import java.util.UUID;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.dataframe.Dataframe;

public class DecisionTreeRegressor extends DecisionTreeLearning {

  public DecisionTreeRegressor(int depth) {
    super(depth, RMSE);
  }

  @Override
  protected Double computePredictedResponse(INDArray array) {
    return array.mean().getDouble(0, 0);
  }

  @Override
  protected Object getNodePrediction(Node node) {
    return node.predictedResponse().doubleValue();
  }

  @Override
  protected List<?> getResponseValues(Dataframe dataframe) {
    return dataframe.get(responseVariableName);
  }

  @Override
  protected INDArray buildY(Dataframe dataframe) {
    var columnTemp = UUID.randomUUID().toString();
    return dataframe.toMatrix(responseVariableName);
  }
}
