package org.rsultan.regression;

import org.apache.commons.math3.distribution.TDistribution;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframe.Column;
import org.rsultan.utils.Matrices;

import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;

public class LinearRegression implements Regression {

    private INDArray BETA;
    private Double R2;
    private INDArray SSR;
    private INDArray SStot;
    private double MSE;
    private double RMSE;
    private INDArray XtXi;
    private INDArray tValues;
    private INDArray pValues;
    private String responseVariableName;
    private String[] predictorNames;


    public LinearRegression setResponseVariableName(String name) {
        this.responseVariableName = name;
        return this;
    }

    public LinearRegression setPredictionVariableName(String name) {
        this.responseVariableName = name;
        return this;
    }

    public LinearRegression setPredictorNames(String... names) {
        this.predictorNames = names;
        return this;
    }

    @Override
    public LinearRegression train(INDArray X, INDArray Y) {
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
        System.out.println("Prediction:");
        Dataframe.create(
                new Column("Predictors", stream(BETA.getColumn(0).toDoubleVector()).boxed().collect(toList())),
                new Column("T-values", stream(tValues.getColumn(0).toDoubleVector()).boxed().collect(toList())),
                new Column("P-values", stream(pValues.getColumn(0).toDoubleVector()).boxed().collect(toList()))
        ).show(BETA.rows());
        System.out.println("");
        Dataframe.create(
                new Column("MSE", List.of(MSE)),
                new Column("RMSE", List.of(RMSE)),
                new Column("R2", List.of(R2))
        ).show(1);

        return this;
    }

    @Override
    public INDArray predict(INDArray X) {
        return BETA.mmul(X);
    }

    private INDArray computeTValues(double degreesOfFreedom) {
        var SE = computeStandardError(degreesOfFreedom);
        return this.BETA.sub(Matrices.average(BETA)).div(SE);
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
        var YMean = Matrices.average(Y);
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