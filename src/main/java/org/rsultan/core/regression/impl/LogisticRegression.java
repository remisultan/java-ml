package org.rsultan.core.regression.impl;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

import java.util.List;
import java.util.function.Function;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.core.regression.GradientDescentRegression;
import org.rsultan.core.regularization.Regularization;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;

public class LogisticRegression extends GradientDescentRegression {

  private static final String YES = "Y";
  private static final String NO = "N";
  protected INDArray YMatrix;
  protected INDArray YoneHot;
  protected List<String> labels;
  private String chosenLabel;

  public LogisticRegression(int numbersOfIterations, double alpha) {
    super(numbersOfIterations, alpha);
  }

  @Override
  public LogisticRegression setResponseVariableName(String name) {
    super.setResponseVariableName(name);
    return this;
  }

  @Override
  public LogisticRegression setPredictionColumnName(String name) {
    super.setPredictionColumnName(name);
    return this;
  }

  @Override
  public LogisticRegression setPredictorNames(String... names) {
    super.setPredictorNames(names);
    return this;
  }

  @Override
  public LogisticRegression setRegularization(Regularization regularization) {
    super.setRegularization(regularization);
    return this;
  }

  @Override
  public LogisticRegression setLossAccuracyOffset(int lossAccuracyOffset) {
    super.setLossAccuracyOffset(lossAccuracyOffset);
    return this;
  }

  public LogisticRegression setChosenLabel(String chosenLabel) {
    this.chosenLabel = chosenLabel;
    return this;
  }

  @Override
  public LogisticRegression setLambda(double lambda) {
    super.setLambda(lambda);
    return this;
  }

  @Override
  public INDArray computeNullHypothesis(INDArray X, INDArray W) {
    return computeSigmoid(X.mmul(W));
  }

  private INDArray computeSigmoid(INDArray z) {
    return sigmoid(z);
  }

  @Override
  protected INDArray computeGradient() {
    var prediction = computeNullHypothesis(X, W);
    return Xt.div(X.rows()).mmul(prediction.sub(YoneHot));
  }

  @Override
  public double computeLoss(INDArray prediction) {
    var h0 = YMatrix.mul(prediction);
    var h1 = YMatrix.neg().add(1).mul(prediction.neg().add(1));
    return h0.add(h1).mean().getDouble(0, 0);
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var dataframeIntercept = dataframe.map(INTERCEPT, () -> 1);
    var X = dataframeIntercept.toMatrix(predictorNames);
    var predictions = computeNullHypothesis(X, W);
    var predictionList = range(0, predictions.rows()).boxed()
        .map(predictions::getRow)
        .map(row -> Nd4j.argMax(row).getInt(0))
        .map(labels::get)
        .map(this.formatPredictedLabel())
        .collect(toList());
    var columns = new Column<>(predictionColumnName, predictionList);
    return dataframe.addColumn(columns);
  }

  protected Function<String, String> formatPredictedLabel() {
    return label -> label.equals(YES) ? this.chosenLabel : "Not " + this.chosenLabel;
  }

  @Override
  public LogisticRegression train(Dataframe dataframe) {
    var df = dataframe
        .map(INTERCEPT, () -> 1)
        .map(this.chosenLabel, obj -> obj.toString().equals(this.chosenLabel) ? YES : NO, responseVariableName);

    X = df.toMatrix(predictorNames);
    XMean = X.mean(true, 1);
    X = X.div(XMean);
    Xt = X.transpose();

    labels = df.get(this.chosenLabel).stream().sorted().distinct().map(Object::toString).collect(toList());

    YoneHot = df.oneHotEncode(this.chosenLabel).select(NO, YES).toMatrix();
    Y = YoneHot.argMax(1).castTo(DataType.DOUBLE);
    W = Nd4j.ones(X.columns(), YoneHot.columns());

    YMatrix = Nd4j.create(Y.toDoubleVector(), Y.columns(), 1);

    this.run();
    return this;
  }
}
