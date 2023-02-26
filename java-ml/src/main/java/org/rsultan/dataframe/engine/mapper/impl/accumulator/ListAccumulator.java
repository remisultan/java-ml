package org.rsultan.dataframe.engine.mapper.impl.accumulator;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public class ListAccumulator implements Accumulator<Object, List<Object>> {

  private final List<Object> list = new ArrayList<>();

  @Override
  public void accumulate(Object value) {
    list.add(value);
  }

  @Override
  public List<Object> get() {
    return list;
  }
}
