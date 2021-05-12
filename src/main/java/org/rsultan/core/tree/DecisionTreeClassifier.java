package org.rsultan.core.tree;

import static java.util.stream.Collectors.toList;
import static org.nd4j.linalg.factory.Nd4j.argMax;

import java.util.List;
import java.util.UUID;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.core.tree.impurity.ImpurityStrategy;
import org.rsultan.dataframe.Dataframe;

public class DecisionTreeClassifier extends DecisionTreeLearning {

  public DecisionTreeClassifier(int depth, ImpurityStrategy strategy) {
    super(depth, strategy);
  }

  @Override
  protected Object getNodePrediction(Node node) {
    return responseValues.get(node.predictedResponse().intValue());
  }

  @Override
  protected List<?> getResponseValues(Dataframe dataframe) {
    return dataframe.get(responseVariableName).stream().sorted().distinct()
        .collect(toList());
  }

  @Override
  protected INDArray buildY(Dataframe dataframe) {
    var columnTemp = UUID.randomUUID().toString();
    return dataframe.map(columnTemp, responseValues::indexOf, responseVariableName)
        .toMatrix(columnTemp);
  }

  @Override
  protected Integer computePredictedResponse(INDArray array) {
    return argMax(array).getInt(0, 0);
  }

  @Override
  public DecisionTreeClassifier train(Dataframe dataframe) {
    super.train(dataframe);
    return this;
  }
}
