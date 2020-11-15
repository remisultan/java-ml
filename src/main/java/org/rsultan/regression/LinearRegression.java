package org.rsultan.regression;

import org.apache.commons.math3.distribution.TDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.utils.Matrices;

import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;

public class LinearRegression implements Regression {

    public static final String INTERCEPT = "Intercept";
    private INDArray BETA;
    private Double R2;
    private INDArray SSR;
    private INDArray SStot;
    private double MSE;
    private double RMSE;
    private INDArray XtXi;
    private INDArray tValues;
    private INDArray pValues;

    private String responseVariableName = "Y";
    private String[] predictorNames = {};
    private String predictionColumnName = "predictions";

    public LinearRegression setResponseVariableName(String name) {
        this.responseVariableName = name;
        return this;
    }

    public LinearRegression setPredictionColumnName(String name) {
        this.predictionColumnName = name;
        return this;
    }

    public LinearRegression setPredictorNames(String... names) {
        String[] strings = {INTERCEPT};
        this.predictorNames = Stream.of(strings, names).flatMap(Arrays::stream).distinct().toArray(String[]::new);
        return this;
    }

    @Override
    public LinearRegression train(Dataframe dataframe) {
        var dataframeIntercept = dataframe.withColumn(INTERCEPT, () -> 1);
        var X = dataframeIntercept.toMatrix(predictorNames);
        var Y = dataframeIntercept.toVector(responseVariableName);

        this.BETA = computeBeta(X, Y);

        var epsilon = Y.sub(X.mmul(BETA));
        this.SSR = epsilon.transpose().mmul(epsilon);

        this.MSE = this.SSR.div(Y.rows()).getDouble(0, 0);
        this.RMSE = Math.sqrt(this.MSE);

        this.SStot = computeSStotal(Y);
        this.R2 = computeRSquare();

        double degreesOfFreedom = (double) Y.rows() - (double) BETA.rows();

        this.tValues = computeTValues(degreesOfFreedom);
        this.pValues = computePValues(degreesOfFreedom);

        return this;
    }

    public LinearRegression showMetrics() {
        System.out.println("\nPrediction:");
        Dataframes.create(
                new Column<>("", stream(predictorNames).collect(toList())),
                new Column<>("Predictors", stream(BETA.getColumn(0).toDoubleVector()).boxed().collect(toList())),
                new Column<>("T-values", stream(tValues.getColumn(0).toDoubleVector()).boxed().collect(toList())),
                new Column<>("P-values", stream(pValues.getColumn(0).toDoubleVector()).boxed().collect(toList()))
        ).show(BETA.rows());
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
        var predictions = stream(X.mmul(BETA).toDoubleVector()).boxed().collect(toList());
        var predictionColumn = new Column<>(this.predictionColumnName, predictions);
        return dataframe.addColumn(predictionColumn);
    }

    private INDArray computeTValues(double degreesOfFreedom) {
        var SE = computeStandardError(degreesOfFreedom);
        return this.BETA.div(SE);
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

    public INDArray getBETA() {
        return BETA;
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