package org.rsultan.core.tree;

import static java.util.Map.Entry.comparingByValue;
import static java.util.Objects.isNull;
import static java.util.stream.Collectors.toList;

import java.util.Map;
import java.util.UUID;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.Trainable;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.core.tree.impurity.ImpurityStrategy;
import org.rsultan.dataframe.Dataframe;

public class DecisionTreeClassifier extends DecisionTreeLearning implements
    Trainable<DecisionTreeClassifier> {

  public DecisionTreeClassifier(int depth, ImpurityStrategy strategy) {
    super(depth, strategy);
  }

  @Override
  public DecisionTreeClassifier train(Dataframe dataframe) {
    var dfNoResponse = dataframe.copy().mapWithout(responseVariableName);
    var dfFeatures = isNull(predictorNames) || predictorNames.length == 0 ? dfNoResponse.copy()
        : dfNoResponse.copy().select(predictorNames);
    responses = dataframe.copy().getColumn(responseVariableName).stream().sorted().distinct()
        .collect(toList());
    train(dfFeatures.toMatrix(), buildY(dataframe));
    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var matrixDf = isNull(predictorNames) || predictorNames.length == 0
        ? getPredictDataframe(dataframe)
        : dataframe.copy().select(predictorNames);
    var predictions = this.predict(matrixDf.toMatrix());
    return dataframe.addColumn(predictionColumnName, predictions);
  }

  @Override
  protected Object getNodePrediction(Node node) {
    return responses.get(node.predictedResponse().intValue());
  }

  protected INDArray buildY(Dataframe dataframe) {
    var columnTemp = UUID.randomUUID().toString();
    return dataframe.copy().map(columnTemp, responses::indexOf, responseVariableName)
        .select(columnTemp).toMatrix();
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
  public DecisionTreeClassifier setShuffle(boolean shuffle) {
    super.setShuffle(shuffle);
    return this;
  }
}
