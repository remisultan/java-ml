package org.rsultan.core.tree.impurity;

public enum ImpurityStrategy {
  GINI, ENTROPY, RMSE;

  public ImpurityService getImpurityService(int totalLabels) {
    return switch (this) {
      case ENTROPY -> new EntropyService(totalLabels);
      case GINI -> new GiniService(totalLabels);
      case RMSE -> new RmseService(totalLabels);
    };
  }
}
