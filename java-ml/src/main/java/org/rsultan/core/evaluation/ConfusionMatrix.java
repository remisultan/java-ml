package org.rsultan.core.evaluation;


public record ConfusionMatrix(double tp, double tn, double fp, double fn) {

  public double recall() {
    return getScore(tp / (tp + fn));
  }

  public double selectivity() {
    return getScore(tn / (tn + fp));
  }

  public double precision() {
    return getScore(tp / (tp + fp));
  }


  public double fallout() {
    return getScore(fp / (fp + tn));
  }

  public double f1Score() {
    final double precision = precision();
    final double recall = recall();
    return getScore(2 * precision * recall / (precision + recall));
  }

  public double phiCoefficient() {
    return getScore((tp * tn - fp * fn) / Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)));
  }

  private static double getScore(double fallOut) {
    return Double.isNaN(fallOut) ? 0d : fallOut;
  }
}
