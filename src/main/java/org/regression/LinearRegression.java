package org.regression;

import org.apache.commons.math3.distribution.TDistribution;
import org.apache.commons.math3.stat.inference.TTest;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.utils.Matrices;

import java.util.stream.DoubleStream;

import static org.apache.commons.lang3.ArrayUtils.toPrimitive;

public class LinearRegression implements Regression {

    private INDArray BETA;
    private INDArray E;
    private Double R2;
    private INDArray SSR;
    private INDArray SStot;
    private INDArray MSE;
    private INDArray RMSE;
    private INDArray XtXi;
    private INDArray varianceMatrix;
    private INDArray SE;
    private INDArray tValues;
    private INDArray pValues;

    @Override
    public LinearRegression train(INDArray X, INDArray Y) {
        this.BETA = computeBeta(X, Y);
        this.E = Y.sub(X.mmul(BETA));
        this.SSR = this.E.transpose().mmul(E);

        this.MSE = this.SSR.div(Y.rows());
        this.RMSE = Transforms.sqrt(this.MSE);

        this.SStot = computeSStotal(Y);
        this.R2 = computeRSquare();

        double degreesOfFreedom = (double) Y.rows() - (double) BETA.rows();
        this.varianceMatrix = XtXi.mul(this.SSR.getDouble(0, 0)).div(degreesOfFreedom);
        this.SE = Transforms.sqrt(Matrices.diagonal(this.varianceMatrix));
        this.tValues = this.BETA.sub(Matrices.average(BETA)).div(this.SE);
        var TDistribution = new TDistribution(degreesOfFreedom);
        this.pValues = Nd4j.create(toPrimitive(
                DoubleStream.of(this.tValues.toDoubleVector()).boxed()
                        .map(TDistribution::cumulativeProbability
                ).toArray(Double[]::new))
        ,tValues.rows(), 1);
        return this;
    }

    @Override
    public INDArray predict(INDArray X) {
        return BETA.mmul(X);
    }

    private Double computeRSquare() {
        var value = SSR.div(this.SStot);
        return 1 - value.getDouble(0, 0);
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

    public INDArray getMSE() {
        return MSE;
    }

    public INDArray getRMSE() {
        return RMSE;
    }

    public INDArray gettValues() {
        return tValues;
    }

    public INDArray getpValues() {
        return pValues;
    }
}