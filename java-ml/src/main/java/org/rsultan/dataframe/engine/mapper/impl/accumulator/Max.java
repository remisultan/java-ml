package org.rsultan.dataframe.engine.mapper.impl.accumulator;

import java.util.Optional;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class Max implements Accumulator<Number, Double> {

  private Double max;

  @Override
  public void accumulate(Number value) {
    final double doubleValue = value.doubleValue();
    if (this.max == null || doubleValue > this.max) {
      this.max = doubleValue;
    }
  }

  @Override
  public Double get() {
    return Optional.ofNullable(max).orElse(Double.NaN);
  }
}
