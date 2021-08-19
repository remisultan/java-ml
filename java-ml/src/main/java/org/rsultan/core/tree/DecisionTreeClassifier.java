package org.rsultan.core.tree;

import static java.util.Arrays.stream;
import static java.util.Map.Entry.comparingByValue;
import static java.util.stream.Collectors.toList;

import java.util.Map;
import java.util.UUID;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.Trainable;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.core.tree.impurity.ImpurityStrategy;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;

public class DecisionTreeClassifier extends DecisionTreeLearning implements
    Trainable<DecisionTreeClassifier> {

  public DecisionTreeClassifier(int depth, ImpurityStrategy strategy) {
    super(depth, strategy);
  }

  @Override
  public DecisionTreeClassifier train(Dataframe dataframe) {
    var dfNoResponse = dataframe.mapWithout(responseVariableName);
    var dfFeatures =
        predictorNames.length == 0 ? dfNoResponse : dfNoResponse.select(predictorNames);
    features = stream(dfFeatures.getColumns()).map(Column::columnName).collect(toList());
    responses = dataframe.get(responseVariableName).stream().sorted().distinct()
        .collect(toList());
    train(dfFeatures.toMatrix(), buildY(dataframe));
    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var predictions = new Column<>(predictionColumnName,
        this.predict(dataframe.getRowSize(), dataframe.select(predictorNames)));
    return dataframe.addColumn(predictions);
  }

  @Override
  protected Object getNodePrediction(Node node) {
    return responses.get(node.predictedResponse().intValue());
  }

  protected INDArray buildY(Dataframe dataframe) {
    var columnTemp = UUID.randomUUID().toString();
    return dataframe.map(columnTemp, responses::indexOf, responseVariableName).toMatrix(columnTemp);
  }

  @Override
  protected Double computePredictedResponse(INDArray array) {
    return impurityService.getClassCount(array).entrySet().stream()
        .max(comparingByValue()).orElse(Map.entry(0D, 0L))
        .getKey();
  }

  @Override
  public DecisionTreeClassifier setResponseVariableName(String responseVariableName) {
    super.setResponseVariableName(responseVariableName);
    return this;
  }

  @Override
  public DecisionTreeClassifier setPredictionColumnName(String name) {
    super.setPredictionColumnName(name);
    return this;
  }

  @Override
  public DecisionTreeClassifier setPredictorNames(String... names) {
    super.setPredictorNames(names);
    return this;
  }

  @Override
  protected Object getPredictionNodeFeatureName(Node node) {
    return features.get(node.feature());
  }
}
