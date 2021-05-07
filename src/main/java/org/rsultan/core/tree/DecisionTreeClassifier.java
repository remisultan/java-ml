package org.rsultan.core.tree;

import static java.util.Arrays.stream;
import static java.util.Comparator.comparing;
import static java.util.Map.Entry.comparingByValue;
import static java.util.Map.entry;
import static java.util.function.Function.identity;
import static java.util.stream.Collectors.counting;
import static java.util.stream.Collectors.groupingBy;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static org.nd4j.linalg.factory.Nd4j.argMax;
import static org.rsultan.core.tree.impurity.ImpurityStrategy.ENTROPY;

import java.io.IOException;
import java.util.List;
import java.util.UUID;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.ModelParameters;
import org.rsultan.core.Trainable;
import org.rsultan.core.tree.domain.Node;
import org.rsultan.core.tree.impurity.ImpurityService;
import org.rsultan.core.tree.impurity.ImpurityStrategy;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;

public class DecisionTreeClassifier
    extends ModelParameters<DecisionTreeClassifier>
    implements Trainable<DecisionTreeClassifier> {

  private final int depth;
  private final ImpurityStrategy strategy;
  private List<String> labelNames;
  private Node tree;
  private ImpurityService impurityService;
  private List<String> featuresNames;

  public DecisionTreeClassifier(int depth, ImpurityStrategy strategy) {
    this.depth = depth;
    this.strategy = strategy;
  }

  public static void main(String[] args) throws IOException {
    Nd4j.setDefaultDataTypes(DataType.DOUBLE, DataType.DOUBLE);
    var dtl = new DecisionTreeClassifier(5, ENTROPY);
    dtl.setResponseVariableName("c4")
        .train(
            Dataframes
                .csv("C:\\Users\\33646\\dev\\nd4j-ml\\src\\main\\resources\\softmax\\iris.data",
                    ",", "\"", false)
        );
  }

  @Override
  public DecisionTreeClassifier train(Dataframe dataframe) {
    var dataframeNoResponse = dataframe.mapWithout(responseVariableName);
    var X = dataframeNoResponse.toMatrix(predictorNames);
    featuresNames = stream(dataframeNoResponse.getColumns()).map(Column::columnName)
        .collect(toList());
    labelNames = dataframe.get(responseVariableName).stream()
        .sorted().distinct().map(String::valueOf)
        .collect(toList());
    impurityService = strategy.getImpurityService(labelNames.size());
    var columnTemp = UUID.randomUUID().toString();
    var Y = dataframe.map(columnTemp, labelNames::indexOf, responseVariableName)
        .toMatrix(columnTemp);
    this.tree = buildTree(X, Y, depth);
    return this;
  }

  private Node buildTree(INDArray features, INDArray response, int currentDepth) {
    if (currentDepth < 1) {
      return null;
    }

    var predictedLabel = stream(response.toIntVector()).parallel().boxed()
        .collect(groupingBy(identity(), counting()))
        .entrySet().stream()
        .max(comparingByValue())
        .orElse(entry(0, 0L))
        .getKey();

    var impurity = impurityService.compute(response).getDouble(0, 0);
    var sortedLabels = range(0, features.columns())
        .mapToObj(col -> features.getColumn(col, false))
        .map(col -> Nd4j.create(col.toDoubleVector(), features.rows(), 1))
        .map(column -> Nd4j.concat(1, column, response))
        .map(matrix -> Nd4j.sortRows(matrix, 0, true))
        .map(matrix -> Nd4j.create(matrix.getColumn(1).toDoubleVector(), features.rows(), 1))
        .collect(toList());
    var allLabels = Nd4j.create(sortedLabels, features.columns(), features.rows()).transpose();

    final double m = allLabels.rows();
    var gain = range(1, (int) m).parallel().mapToObj(i -> {
      var left = allLabels.getRows(range(0, i).toArray());
      var right = allLabels.getRows(range(i, (int) m).toArray());
      var impurityLeft = impurityService.compute(left).mul(i / m);
      var impurityRight = impurityService.compute(right).mul((m - i) / m);
      var impurities = impurityLeft.add(impurityRight).neg().add(impurity);
      return new GainHolder(i, argMax(impurities).getInt(0, 0), impurities.max().getDouble(0, 0));
    }).max(comparing(GainHolder::gain)).orElse(new GainHolder(0, 0, 0));

    var rangeLeft = range(1, gain.split).toArray();
    var rangeRight = range(gain.split, features.rows()).toArray();
    return new Node(
        response.getInt(gain.label),
        features.getDouble(gain.split, gain.label),
        predictedLabel,
        buildNode(features, response, currentDepth, rangeLeft),
        buildNode(features, response, currentDepth, rangeRight)
    );
  }

  private Node buildNode(INDArray features, INDArray response, int currentDepth, int[] rangeLeft) {
    return rangeLeft.length == 0 ? null : this
        .buildTree(features.getRows(rangeLeft), response.getRows(rangeLeft), currentDepth - 1);
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    return null;
  }

  private static record GainHolder(int split, int label, double gain) {

  }
}
