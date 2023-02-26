package org.rsultan.dataframe.engine.mapper.impl.accumulator;

import java.util.Optional;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class Min implements Accumulator<Number, Double> {

  private Double min;

  @Override
  public void accumulate(Number value) {
    final double doubleValue = value.doubleValue();
    if (this.min == null || doubleValue < this.min) {
      this.min = doubleValue;
    }
  }

  @Override
  public Double get() {
    return Optional.ofNullable(min).orElse(Double.NaN);
  }
}
