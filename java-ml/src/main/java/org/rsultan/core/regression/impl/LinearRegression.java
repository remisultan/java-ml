package org.rsultan.core.regression.impl;

import java.util.ArrayList;
import org.apache.commons.math3.distribution.TDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.rsultan.core.regression.Regression;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.core.regression.AbstractRegression;
import org.rsultan.utils.Matrices;

import java.util.List;
import java.util.stream.DoubleStream;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;

public class LinearRegression extends AbstractRegression {

  private Double R2;
  private INDArray SSR;
  private INDArray SStot;
  private double MSE;
  private double RMSE;
  private INDArray XtXi;
  private INDArray tValues;
  private INDArray pValues;

  public LinearRegression setResponseVariableName(String name) {
    super.setResponseVariableName(name);
    return this;
  }

  public LinearRegression setPredictionColumnName(String name) {
    super.setPredictionColumnName(name);
    return this;
  }

  public LinearRegression setPredictorNames(String... names) {
    super.setPredictorNames(names);
    return this;
  }

  public LinearRegression setShuffle(boolean shuffle) {
    super.setShuffle(shuffle);
    return this;
  }

  @Override
  public LinearRegression train(Dataframe dataframe) {
    var dataframeIntercept = dataframe.copy().map(INTERCEPT, () -> 1).select(predictorNames);
    X = dataframeIntercept.toMatrix();
    Y = dataframe.copy().select(responseVariableName).toMatrix();

    shuffle(X, Y);

    Xt = X.transpose();
    this.W = computeBeta(X, Y);
    this.RMSE = computeLoss(computeNullHypothesis(X, W));
    this.SStot = computeSStotal(Y);
    this.R2 = computeRSquare();

    double degreesOfFreedom = (double) Y.rows() - (double) W.rows();

    this.tValues = computeTValues(degreesOfFreedom);
    this.pValues = computePValues(degreesOfFreedom);

    return this;
  }

  public LinearRegression showMetrics() {
    System.out.println("\nPrediction:");
    var rows = new ArrayList<List<?>>(W.rows());
    for (int i = 0; i < W.rows(); i++) {
      rows.add(List.of(predictorNames[i], W.getScalar(i, 0), tValues.getScalar(i, 0),
          pValues.getScalar(i, 0)));
    }
    Dataframes.create(new String[]{"", "Predictors", "T-values", "P-values"}, rows).show(W.rows());
    System.out.println();
    Dataframes.create(new String[]{"MSE", "RMSE", "R2"}, List.of(List.of(MSE, RMSE, R2))).show(1);
    return this;
  }

  @Override
  public Dataframe predict(Dataframe dataframe) {
    var dataframeIntercept = dataframe.copy().map(INTERCEPT, () -> 1).select(this.predictorNames);
    var X = dataframeIntercept.toMatrix();
    List<Object> predictions = DoubleStream.of(computeNullHypothesis(X, W).toDoubleVector()).boxed()
        .collect(toList());
    return dataframe.copy().addColumn(this.predictionColumnName, predictions);
  }

  @Override
  public INDArray computeNullHypothesis(INDArray X, INDArray W) {
    return X.mmul(W);
  }

  @Override
  public double computeLoss(INDArray prediction) {
    var epsilon = Y.sub(prediction);
    this.SSR = epsilon.transpose().mmul(epsilon);

    this.MSE = this.SSR.div(Y.rows()).getDouble(0, 0);
    return Math.sqrt(this.MSE);
  }

  private INDArray computeTValues(double degreesOfFreedom) {
    var SE = computeStandardError(degreesOfFreedom);
    return W.div(SE);
  }

  private INDArray computeStandardError(double degreesOfFreedom) {
    var varianceMatrix = XtXi.mul(this.SSR.getDouble(0, 0)).div(degreesOfFreedom);
    return Transforms.sqrt(Matrices.diagonal(varianceMatrix));
  }

  private INDArray computePValues(double degreesOfFreedom) {
    var tDist = new TDistribution(degreesOfFreedom);
    double[] tValues = this.tValues.toDoubleVector();
    double[] pValues = DoubleStream.of(tValues).map(tDist::cumulativeProbability).map(p -> 1 - p)
        .toArray();
    return Nd4j.create(pValues, this.tValues.rows(), 1);
  }

  private Double computeRSquare() {
    var rSquare = SSR.div(this.SStot);
    return 1 - rSquare.getDouble(0, 0);
  }

  private INDArray computeSStotal(INDArray Y) {
    var YMean = Matrices.vectorAverage(Y);
    var yDemeaned = Y.sub(YMean);
    return yDemeaned.transpose().mmul(yDemeaned);
  }

  private INDArray computeBeta(INDArray X, INDArray Y) {
    var XtX = Xt.mmul(X);
    var XtY = Xt.mmul(Y);
    XtXi = InvertMatrix.invert(XtX, false);
    return XtXi.mmul(XtY);
  }

  public INDArray getW() {
    return W;
  }

  public Double getR2() {
    return R2;
  }

  public double getMSE() {
    return MSE;
  }

  public double getRMSE() {
    return RMSE;
  }

  public INDArray gettValues() {
    return tValues;
  }

  public INDArray getpValues() {
    return pValues;
  }
}