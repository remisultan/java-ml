package org.rsultan.dataframe.engine.mapper;

public interface Mapper<T, R> {

  R map(T element);

}
