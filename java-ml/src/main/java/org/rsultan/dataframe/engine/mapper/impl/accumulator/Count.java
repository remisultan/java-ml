package org.rsultan.dataframe.engine.mapper.impl.accumulator;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class Count implements Accumulator<Number, Long> {

  private Long count = 0L;

  @Override
  public void accumulate(Number value) {
    count++;
  }

  @Override
  public Long get() {
    return count;
  }
}
