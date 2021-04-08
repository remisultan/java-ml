package org.rsultan.dataframe.transform.matrix;

import static java.lang.Double.parseDouble;
import static java.util.Arrays.stream;
import static java.util.Comparator.comparing;
import static java.util.function.Predicate.not;
import static java.util.stream.Collectors.toMap;
import static java.util.stream.Stream.of;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.rsultan.dataframe.Column;
import org.rsultan.dataframe.Dataframe;
import org.rsultan.dataframe.Dataframes;

public class MatrixDataframe implements MatrixTransform {

  private static final String NUMBER_REGEX = "^\\d+(\\.\\d+)*$";
  private final Dataframe dataframe;

  public MatrixDataframe(Dataframe dataframe) {
    this.dataframe = dataframe;
  }

  public INDArray toVector(String columnName) {
    double[] doubles = this.dataframe.getData().get(columnName).stream()
        .parallel()
        .mapToDouble(obj -> parseDouble(obj.toString())).toArray();
    return Nd4j.create(doubles, doubles.length, 1);
  }

  public INDArray toMatrix(String... columnNames) {
    var colNameStream = columnNames.length == 0 ?
        stream(this.dataframe.getColumns()).map(Column::values) :
        of(columnNames).map(this.dataframe::get);

    var vectorList = colNameStream.parallel()
        .map(List::stream)
        .map(valueStream -> valueStream.mapToDouble(this::objectToDouble))
        .map(DoubleStream::toArray)
        .map(doubles -> Nd4j.create(doubles, doubles.length, 1))
        .toArray(INDArray[]::new);
    return Nd4j.concat(1, vectorList);
  }

  @Override
  public Dataframe oneHotEncode(String columnToEncode) {
    var toEncodeColumnMap = this.dataframe
        .get(columnToEncode).stream()
        .distinct().sorted()
        .map(colName -> new Column<Boolean>(colName.toString(), new ArrayList<>()))
        .collect(toMap(Column::columnName, c -> c));

    this.dataframe.get(columnToEncode)
        .parallelStream()
        .map(Object::toString)
        .forEachOrdered(colName -> {
          var trueCol = toEncodeColumnMap.get(colName);
          trueCol.values().add(true);
          toEncodeColumnMap.keySet().parallelStream()
              .map(toEncodeColumnMap::get)
              .filter(not(trueCol::equals))
              .forEachOrdered(column -> column.values().add(false));
        });

    var columnArray = toEncodeColumnMap.values()
        .stream()
        .sorted(comparing(Column::columnName))
        .toArray(Column[]::new);

    return Dataframes.create(
        of(this.dataframe.getColumns(), columnArray).flatMap(Arrays::stream).toArray(Column[]::new)
    );
  }

  private Double objectToDouble(Object obj) {
    if (obj instanceof Number number) {
      return number.doubleValue();
    } else if (obj instanceof Boolean b) {
      return b ? 1.0D : 0.0D;
    } else if (obj instanceof String s && s.trim().matches(NUMBER_REGEX)) {
      return parseDouble(s.trim());
    }
    throw new IllegalArgumentException("Cannot cast " + obj + " to number");
  }
}
