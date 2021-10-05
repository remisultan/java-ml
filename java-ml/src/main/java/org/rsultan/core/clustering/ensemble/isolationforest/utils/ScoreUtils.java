package org.rsultan.core.clustering.ensemble.isolationforest.utils;

public class ScoreUtils {

  private static final double EULER_CONSTANT = 0.5772156649;

  public static double averagePathLength(double leafSize) {
    if (leafSize > 2) {
      return 2 * harmonicNumber(leafSize) - (2 * (leafSize - 1) / leafSize);
    }
    if (leafSize == 2) {
      return 1;
    }
    return 0;
  }

  private static double harmonicNumber(double leafSize) {
    return Math.log(leafSize - 1) + EULER_CONSTANT;
  }
}
