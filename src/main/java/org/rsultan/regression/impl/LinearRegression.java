package org.rsultan.regression.impl;

import org.apache.commons.math3.distribution.TDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.regression.AbstractRegression;
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

    @Override
    public LinearRegression train(Dataframe dataframe) {
        var dataframeIntercept = dataframe.withColumn(INTERCEPT, () -> 1);
        X = dataframeIntercept.toMatrix(predictorNames);
        Xt = dataframeIntercept.toMatrix(predictorNames);
        XMean = X.mean(true, 1);
        Y = dataframeIntercept.toVector(responseVariableName);

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
        Dataframes.create(
                new Column<>("", stream(predictorNames).collect(toList())),
                new Column<>("Predictors", stream(W.getColumn(0).toDoubleVector()).boxed().collect(toList())),
                new Column<>("T-values", stream(tValues.getColumn(0).toDoubleVector()).boxed().collect(toList())),
                new Column<>("P-values", stream(pValues.getColumn(0).toDoubleVector()).boxed().collect(toList()))
        ).show(W.rows());
        System.out.print("\n");
        Dataframes.create(
                new Column<>("MSE", List.of(MSE)),
                new Column<>("RMSE", List.of(RMSE)),
                new Column<>("R2", List.of(R2))
        ).show(1);

        return this;
    }

    @Override
    public Dataframe predict(Dataframe dataframe) {
        var dataframeIntercept = dataframe.withColumn(INTERCEPT, () -> 1);
        var X = dataframeIntercept.toMatrix(this.predictorNames);
        var prediction = computeNullHypothesis(X, W);
        var predictions = stream(prediction.toDoubleVector()).boxed().collect(toList());
        var predictionColumn = new Column<>(this.predictionColumnName, predictions);
        return dataframe.addColumn(predictionColumn);
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
        double[] pValues = DoubleStream.of(tValues).map(tDist::cumulativeProbability).toArray();
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
        var Xt = X.transpose();
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