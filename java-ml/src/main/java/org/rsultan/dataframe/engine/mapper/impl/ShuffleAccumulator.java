package org.rsultan.dataframe.engine.mapper.impl;

import java.util.Collections;
import org.rsultan.dataframe.Row;
import org.rsultan.dataframe.engine.mapper.AccumulatorDataProcessor;

public class ShuffleAccumulator extends AccumulatorDataProcessor {

  @Override
  public Row map(Row element) {
    return element;
  }

  @Override
  protected void feedFromAccumulator() {
    Collections.shuffle(this.accumulator);
    super.feedFromAccumulator();
  }
}
