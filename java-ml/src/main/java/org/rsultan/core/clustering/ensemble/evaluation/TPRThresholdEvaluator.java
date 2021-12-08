package org.rsultan.core.clustering.ensemble.evaluation;

import org.rsultan.core.Evaluator;
import org.rsultan.core.clustering.ensemble.isolationforest.IsolationForest;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.TrainTestDataframe;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TPRThresholdEvaluator implements Evaluator<Double, IsolationForest> {

    private static final Logger LOG = LoggerFactory.getLogger(TPRThresholdEvaluator.class);
    private static final String IS_AN_ANOMALY = "isAnAnomaly";
    private final String responseVariable;
    private final String predictionColumn;
    private double desiredTPR = 0.9;
    private double learningRate = 0.01;
    private double TPR = 0;
    private double FPR = 0;
    private Dataframe externalTestDataframe;

    public TPRThresholdEvaluator(String responseVariable, String predictionColumn) {
        this.responseVariable = responseVariable;
        this.predictionColumn = predictionColumn;
    }

    @Override
    public Double evaluate(IsolationForest trainable, TrainTestDataframe dataframe) {
        var dfSplit = dataframe.shuffle().split();
        var trained = trainable.setUseAnomalyScoresOnly(true)
                .train(dfSplit.train().mapWithout(responseVariable));
        final Dataframe testDf = externalTestDataframe != null ? externalTestDataframe : dfSplit.test();
        final Dataframe predict = trained.predict(testDf.mapWithout(responseVariable));

        double threshold = 1;
        while (threshold > 0 && TPR <= desiredTPR) {
            LOG.info("Evaluating isolation forest with threshold {}", threshold);
            threshold -= learningRate;
            final double finalThreshold = threshold;
            var responses = testDf.<Long>get(responseVariable);
            var predictions = predict.map(IS_AN_ANOMALY,
                            (Double score) -> (score >= finalThreshold ? 1L : 0L), predictionColumn)
                    .<Long>get(IS_AN_ANOMALY);

            double truePositives = 0;
            double trueNegatives = 0;
            double falsePositives = 0;
            double falseNegative = 0;

            for (int i = 0; i < responses.size(); i++) {
                var response = responses.get(i);
                var prediction = predictions.get(i);
                truePositives += response == 1L && prediction == 1L ? 1L : 0L;
                trueNegatives += response == 0L && prediction == 0L ? 1L : 0L;
                falsePositives += response == 0L && prediction == 1L ? 1L : 0L;
                falseNegative += response == 1L && prediction == 0L ? 1L : 0L;
            }
            TPR = truePositives / (truePositives + falseNegative);
            TPR = Double.isNaN(TPR) ? 0 : TPR;

            FPR = falsePositives / (falsePositives + trueNegatives);
            FPR = Double.isNaN(FPR) ? 0L : FPR;
        }
        if (threshold < 0) {
            throw new IllegalArgumentException("Cannot have desired TPR");
        }

        return threshold;
    }

    public TPRThresholdEvaluator setDesiredTPR(double desiredTPR) {
        this.desiredTPR = desiredTPR;
        return this;
    }

    public TPRThresholdEvaluator setLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public TPRThresholdEvaluator setExternalTestDataframe(Dataframe dataframe) {
        this.externalTestDataframe = dataframe;
        return this;
    }

    public void showMetrics() {
        Dataframes.create(new String[]{"TPR", "FPR"}, new Row(TPR, FPR)).tail();
    }
}
