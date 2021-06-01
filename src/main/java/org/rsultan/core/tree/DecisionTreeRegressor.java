package org.rsultan.core.tree;

import static org.rsultan.core.tree.impurity.ImpurityStrategy.RMSE;

import java.util.List;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.dataframe.Dataframe;

public class DecisionTreeRegressor extends DecisionTreeLearning {

  public DecisionTreeRegressor(int depth) {
    super(depth, RMSE);
  }

  @Override
  public DecisionTreeRegressor train(Dataframe dataframe) {
    super.train(dataframe);
    return this;
  }

  @Override
  public DecisionTreeRegressor setResponseVariableName(String responseVariableName) {
    super.setResponseVariableName(responseVariableName);
    return this;
  }

  @Override
  public DecisionTreeRegressor setPredictionColumnName(String name) {
    super.setPredictionColumnName(name);
    return this;
  }

  @Override
  public DecisionTreeRegressor setPredictorNames(String... names) {
    super.setPredictorNames(names);
    return this;
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
    return dataframe.toMatrix(responseVariableName);
  }

  @Override
  protected Object getPredictionNodeFeatureName(Node node) {
    return features.get(node.feature());
  }
}
