package org.rsultan.regression.impl;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.regression.GradientDescentRegression;
import org.rsultan.regularization.Regularization;

import java.util.List;
import java.util.function.Function;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

public class LogisticRegression extends GradientDescentRegression {

    private static final String YES = "Y";
    private static final String NO = "N";
    private String label;
    protected INDArray YMatrix;
    protected INDArray YoneHot;
    protected List<String> labels;

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

    public LogisticRegression setLabel(String label) {
        this.label = label;
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
        var dataframeIntercept = dataframe.withColumn(INTERCEPT, () -> 1);
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
        return label -> label.equals(YES) ? this.label : "Not " + this.label;
    }

    @Override
    public LogisticRegression train(Dataframe dataframe) {
        var df = dataframe
                .withColumn(this.label, responseVariableName, obj -> obj.toString().equals(this.label) ? YES : NO)
                .withColumn(INTERCEPT, () -> 1);

        X = df.toMatrix(predictorNames);
        XMean = X.mean(true, 1);
        X = X.div(XMean);
        Xt = X.transpose();

        labels = df.get(this.label).stream().distinct().sorted().map(Object::toString).collect(toList());

        YoneHot = df.oneHotEncode(this.label).select(YES, NO).toMatrix();
        Y = YoneHot.argMax(1).castTo(DataType.DOUBLE);
        W = Nd4j.ones(X.columns(), YoneHot.columns());

        YMatrix = Nd4j.create(Y.toDoubleVector(), Y.columns(), 1);

        this.run();
        return this;
    }
}
