package org.rsultan.dataframe.transform.shuffle;

import java.io.Serializable;
import org.rsultan.dataframe.Dataframe;

public interface ShuffleTransform<T extends Dataframe> extends Serializable {

  T shuffle();

}
