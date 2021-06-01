package org.rsultan.core.serde;

import java.io.IOException;

public interface Serialiser<S, T> {

  T serialise(S source) throws IOException;

}
