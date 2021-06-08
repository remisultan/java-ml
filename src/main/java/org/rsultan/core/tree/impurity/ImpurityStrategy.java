package org.rsultan.core.tree.impurity;

public enum ImpurityStrategy {
  GINI, ENTROPY, RMSE;

  public ImpurityService getImpurityService() {
    return switch (this) {
      case ENTROPY -> new EntropyService();
      case GINI -> new GiniService();
      case RMSE -> new RmseService();
    };
  }
}
