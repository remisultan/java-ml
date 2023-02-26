package org.rsultan.dataframe.engine.mapper.impl.accumulator;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class Average implements Accumulator<Number, Double> {

  private double sum = 0D;
  private long count = 0L;

  @Override
  public void accumulate(Number value) {
    sum += value.doubleValue();
    count++;
  }

  @Override
  public Double get() {
    return sum / count;
  }
}
