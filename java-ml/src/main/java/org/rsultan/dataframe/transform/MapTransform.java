package org.rsultan.dataframe.transform;

import java.io.Serializable;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframe.Result;

public interface MapTransform extends Serializable {

  <S, T> Dataframe transform(String columnName, Function<S, T> f);

  <S1, S2, T> Dataframe transform(String columnName, String columnName2, BiFunction<S1, S2, T> f);

  <T> Dataframe map(String columnName, Supplier<T> f);

  Dataframe addColumn(Object columnName, List<?> values);

  <S, T> Dataframe map(String columnName, Function<S, T> f, String sourceColumn);

  <S1, S2, T> Dataframe map(
      String columnName,
      BiFunction<S1, S2, T> transform,
      String sourceColumn1,
      String sourceColumn2
  );

  Dataframe mapWithout(String... columnNames);

  Result getResult();

  <T> List<T> getColumn(Object columnName);
}
