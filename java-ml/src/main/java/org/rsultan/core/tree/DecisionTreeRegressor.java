package org.rsultan.core.tree;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;
import static org.rsultan.core.tree.impurity.ImpurityStrategy.RMSE;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.Trainable;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;

public class DecisionTreeRegressor extends DecisionTreeLearning implements
    Trainable<DecisionTreeRegressor> {

  public DecisionTreeRegressor(int depth) {
    super(depth, RMSE);
  }

  @Override
  public DecisionTreeRegressor train(Dataframe dataframe) {
    var dfNoResponse = dataframe.mapWithout(responseVariableName);
    var dfFeatures = dfNoResponse.select(predictorNames);
    features = stream(dfFeatures.getColumns()).map(Column::columnName).collect(toList());
    responses = dataframe.get(responseVariableName);
    train(dfFeatures.toMatrix(), dataframe.toMatrix(responseVariableName));
    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var predictions = new Column<>(predictionColumnName,
        this.predict(dataframe.getRowSize(), dataframe.select(predictorNames)));
    return dataframe.addColumn(predictions);
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
  protected Object getPredictionNodeFeatureName(Node node) {
    return features.get(node.feature());
  }
}
