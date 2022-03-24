package org.rsultan.dataframe.engine.mapper;

public interface BiMapper<T, U, R> {

  R map(T element, U element2);
}
