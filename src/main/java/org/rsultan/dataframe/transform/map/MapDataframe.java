package org.rsultan.dataframe.transform.map;

import static java.util.Arrays.stream;
import static java.util.stream.Collectors.toList;
import static java.util.stream.IntStream.range;
import static java.util.stream.Stream.of;

import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;

public class MapDataframe implements MapTransform {

  private final Dataframe dataframe;

  public MapDataframe(Dataframe dataframe) {
    this.dataframe = dataframe;
  }

  @Override
  public <T> Dataframe map(String columnName, Supplier<T> supplier) {
    var values = range(0, this.dataframe.getRows()).boxed()
        .map(num -> supplier.get())
        .collect(toList());
    return this.dataframe.addColumn(new Column<>(columnName, values));
  }

  @Override
  public <S, T> Dataframe map(String columnName, Function<S, T> f, String sourceColumn) {
    List<S> values = this.dataframe.get(sourceColumn);
    Column<T> newColumn = new Column<>(columnName, values.stream().map(f).collect(toList()));
    return  this.dataframe.addColumn(newColumn);
  }

  @Override
  public <S1, S2, T> Dataframe map(
      String columnName,
      BiFunction<S1, S2, T> transform,
      String sourceColumn1,
      String sourceColumn2
  ) {
    List<S1> values1 = this.dataframe.get(sourceColumn1);
    List<S2> values2 = this.dataframe.get(sourceColumn2);
    var targetValues = range(0, values1.size()).parallel().boxed()
        .map(index -> transform.apply(values1.get(index), values2.get(index)))
        .collect(toList());
    var newColumn = new Column[]{new Column<>(columnName, targetValues)};
    return Dataframes.create(
        of(this.dataframe.getColumns(), newColumn).flatMap(Arrays::stream).toArray(Column[]::new)
    );
  }

  @Override
  public Dataframe mapWithout(String... columnNames) {
    var colList = List.of(columnNames);
    return Dataframes.create(
        stream(this.dataframe.getColumns())
            .filter(column -> !colList.contains(column.columnName()))
            .toArray(Column[]::new)
    );
  }
}
