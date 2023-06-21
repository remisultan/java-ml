package org.rsultan.core;


import org.rsultan.dataframe.Dataframe;

public interface Evaluator<V, T extends RawTrainable<T>> {

  V evaluate(T trainable, Dataframe dataframe);

}
