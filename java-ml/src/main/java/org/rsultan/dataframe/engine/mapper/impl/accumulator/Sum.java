package org.rsultan.dataframe.engine.mapper.impl.accumulator;

import java.util.Optional;
import javax.swing.text.html.Option;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class Sum implements Accumulator<Number, Double> {

  private Double sum;

  @Override
  public void accumulate(Number value) {
    if (sum == null) {
      sum = 0.0D;
    }
    sum += value.doubleValue();
  }

  @Override
  public Double get() {
    return Optional.ofNullable(sum).orElse(Double.NaN);
  }
}
