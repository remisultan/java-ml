package org.rsultan.core.tree;

import static java.util.Objects.isNull;
import static org.rsultan.core.tree.impurity.ImpurityStrategy.RMSE;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.Trainable;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.dataframe.Dataframe;

public class DecisionTreeRegressor extends DecisionTreeLearning implements
    Trainable<DecisionTreeRegressor> {

  public DecisionTreeRegressor(int depth) {
    super(depth, RMSE);
  }

  @Override
  public DecisionTreeRegressor train(Dataframe dataframe) {
    final Dataframe dfResponse = dataframe.copy().select(responseVariableName);
    var dfNoResponse = dataframe.copy().mapWithout(responseVariableName);
    var dfFeatures = isNull(predictorNames) || predictorNames.length == 0 ? dfNoResponse.copy()
        : dfNoResponse.copy().select(predictorNames);
    responses = dfResponse.getColumn(responseVariableName);
    train(dfFeatures.toMatrix(), dfResponse.toMatrix());
    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var predictions = super.predict(isNull(predictorNames) || predictorNames.length == 0 ?
        getPredictDataframe(dataframe).toMatrix() :
        dataframe.copy().select(predictorNames).toMatrix());
    return dataframe.addColumn(predictionColumnName, predictions);
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
  public DecisionTreeRegressor setShuffle(boolean shuffle) {
    super.setShuffle(shuffle);
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

}
