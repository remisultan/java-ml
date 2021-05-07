package org.rsultan.core.tree.impurity;

public enum ImpurityStrategy {
  GINI, ENTROPY, VARIANCE;

  public ImpurityService getImpurityService(int totalLabels) {
    return switch (this) {
      case ENTROPY -> new EntropyService(totalLabels);
      default -> throw new IllegalStateException("Unexpected value: " + this);
    };
  }
}
