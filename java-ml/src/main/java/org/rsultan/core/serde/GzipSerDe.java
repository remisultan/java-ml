package org.rsultan.core.serde;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class GzipSerDe<T> implements Serialiser<T, byte[]>, Deserialiser<byte[], T> {

  @Override
  public byte[] serialise(T source) throws IOException {
    var bos = new ByteArrayOutputStream();
    var gos = new GZIPOutputStream(bos);
    var oos = new ObjectOutputStream(gos);
    oos.writeObject(source);
    oos.close();
    return bos.toByteArray();
  }

  @Override
  public T deserialise(byte[] source) throws IOException, ClassNotFoundException {
    var bis = new ByteArrayInputStream(source);
    var gis = new GZIPInputStream(bis);
    var ois = new ObjectInputStream(gis);
    T object = (T) ois.readObject();
    ois.close();
    return object;
  }
}
