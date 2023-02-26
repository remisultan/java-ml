package org.rsultan.dataframe.engine.mapper.impl.accumulator;

/**
 * @author Rémi SULTAN (remi.sultan at graviteesource.com)
 * @author GraviteeSource Team
 */
public interface Accumulator<T, V> {

  void accumulate(T value);

  V get();
}
