package org.rsultan.regression;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;

import java.util.ArrayList;

import static java.util.stream.Collectors.toList;
import static java.util.stream.LongStream.range;
import static org.nd4j.linalg.ops.transforms.Transforms.sigmoid;

public class LogisticRegression extends AbstractLogisticRegression {

    public static final String YES = "Y";
    public static final String NO = "N";
    private String label;

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

    public LogisticRegression setLabel(String label) {
        this.label = label;
        return this;
    }

    @Override
    protected INDArray computeNullHypothesis(INDArray X, INDArray W) {
        return computeSigmoid(X.mmul(W));
    }

    private INDArray computeSigmoid(INDArray z) {
        return sigmoid(z);
    }

    @Override
    protected INDArray computeGradient(INDArray X, INDArray Xt, INDArray W, INDArray labels) {
        var prediction = computeNullHypothesis(X, W);
        return Xt.div(X.rows()).mmul(prediction.sub(labels));
    }

    @Override
    protected double computeLoss(INDArray prediction, INDArray trueLabels) {
        var h0 = trueLabels.mul(prediction);
        var h1 = trueLabels.neg().add(1).mul(prediction.neg().add(1));
        return h0.add(h1).mean().getDouble(0, 0);
    }

    @Override
    public LogisticRegression train(Dataframe dataframe) {
        var df = dataframe
                .withColumn(this.label, responseVariableName, obj -> obj.toString().equals(this.label) ? YES : NO)
                .withColumn(INTERCEPT, () -> 1);

        var X = df.toMatrix(predictorNames);
        XMean = X.mean(true, 1);
        X = X.div(XMean);
        var Xt = X.transpose();

        this.labels = df.get(this.label).stream()
                .distinct().sorted()
                .map(Object::toString)
                .collect(toList());

        var YoneHot = df.oneHotEncode(this.label)
                .withoutColumn(this.label)
                .withoutColumn(responseVariableName)
                .withoutColumn(predictorNames).toMatrix();
        var Y = YoneHot.argMax(1).castTo(DataType.DOUBLE);
        W = Nd4j.ones(X.columns(), YoneHot.columns());
        this.run(X, Xt, Y, YoneHot);
        return this;
    }

    @Override
    protected void run(INDArray X, INDArray Xt, INDArray Y, INDArray YoneHot) {
        var loss = new Column<>(LOSS_COLUMN, new ArrayList<Double>());
        var accuracy = new Column<>(ACCURACY_COLUMN, new ArrayList<Double>());
        var YMatrix = Nd4j.create(Y.toDoubleVector(), Y.columns(), 1);

        range(0, this.numbersOfIterations).map(idx -> {
            var gradAlpha = computeGradient(X, Xt, W, YoneHot).mul(this.alpha);
            W.subi(gradAlpha);
            return idx;
        }).forEach(idx -> {
            if (idx % getLossAccuracyOffset() == 0) {
                var prediction = computeNullHypothesis(X, W);
                loss.values().add(computeLoss(prediction, YMatrix));
                accuracy.values().add(computeAccuracy(X, W, Y));
            }
        });

        this.history = Dataframes.create(loss, accuracy);
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
                .map(label -> label.equals(YES) ? this.label : "Not " + this.label)
                .collect(toList());
        var columns = new Column<>(predictionColumnName, predictionList);
        return dataframe.addColumn(columns);
    }
}
