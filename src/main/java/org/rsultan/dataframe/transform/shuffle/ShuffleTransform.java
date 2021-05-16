package org.rsultan.dataframe.transform.shuffle;

import org.rsultan.dataframe.Dataframe;

public interface ShuffleTransform<T extends Dataframe> {

  T shuffle();

}
