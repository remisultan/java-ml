package org.rsultan.dataframe.transform.map;

import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import org.rsultan.dataframe.Dataframe;

public interface MapTransform {

  <T> Dataframe map(String columnName, Supplier<T> supplier);

  <S, T> Dataframe map(String columnName, Function<S, T> f, String sourceColumn);

  <S1, S2, T> Dataframe map(
      String columnName,
      BiFunction<S1, S2, T> transform,
      String sourceColumn1,
      String sourceColumn2
  );

  Dataframe mapWithout(String... columnNames);
}
